"""
GPU / CPU 效能量測 —— 照課程投影片 4-3「測速三守則」寫成、可以重跑的量測。

    python benchmark_gpu.py                 # 全部項目（約 3~5 分鐘）
    python benchmark_gpu.py --quick         # 快速版（約 1 分鐘），只用來確認流程
    python benchmark_gpu.py --only train,loader

為什麼要有這支腳本
------------------
README 的速度數字（推論 22.7 ms、GPU 訓練快 11.7 倍）是當時臨時量的，
量測程式沒有留下來。2026-09-22 用現行程式重量，num_workers=0 時每批 215 ms，
和 README 記載的 102 ms 對不上，卻已經查不出差在哪裡。
這支腳本把量測方法固定下來：誰都能重跑，每個數字都附硬體、版本、設定與分散度。

三守則（每一項計時都遵守，見 timed()）
--------------------------------------
  1. warm-up       先跑幾次丟掉。第一次呼叫含 CUDA 初始化、kernel 載入與顯存配置
  2. synchronize   計時前後都同步。CUDA 是非同步的，不同步量到的只是「送指令多快」
  3. 重複取中位數  預設 20 次，回報中位數與 IQR（四分位距）

量測項目（對應投影片 4-8 的量測報告表）
--------------------------------------
  env       環境資訊：GPU、顯存、運算能力、CUDA / PyTorch 版本、CPU（一律執行）
  matmul    矩陣乘法 CPU vs GPU，換算加速倍數與 TFLOPS（投影片 LAB 4-1）
  infer     本專案兩個模型的推論：CPU vs GPU、第一次呼叫、批次吞吐、顯存
  pipeline  單張照片全流程的時間拆解：解碼 / 前處理+推論 / 影像量測
            （需要權重與 data/my_photos 內的照片，缺少時略過）
  train     訓練單步：batch size 掃描 x FP32 / 混合精度，含峰值顯存（投影片 4-5、4-6）
  loader    資料管線：DataLoader 的 num_workers 對實際訓練速度的影響（投影片 4-7）
            （需要 data/ava_train.csv 與 data/dataset，缺少時略過）

只量時間，不改任何東西
----------------------
不讀寫任何權重檔（pipeline 只讀取現役權重）。train / loader 的模型是在記憶體裡
臨時建立的，量完即丟。混合精度只做量測，訓練腳本並未採用。

輸出
----
  reports/benchmark/<時間>/benchmark.json   所有數據，含每一次的原始量測值
  reports/benchmark/<時間>/summary.md       表格版摘要
  reports/benchmark/<時間>/*.png            圖表（需要 matplotlib）
"""
import argparse
import json
import os
import platform
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

import torch

BASE_DIR = Path(__file__).resolve().parent
SECTIONS = ('env', 'matmul', 'infer', 'pipeline', 'train', 'loader')
MB = 1024 ** 2

FULL = dict(reps=20, matmul_sizes=(512, 1024, 2048, 4096), infer_batches=(1, 8, 32, 64),
            train_batches=(16, 32, 64, 128, 256), workers=(0, 2, 4, 8),
            loader_windows=5, loader_window_steps=10)
# 快速模式的重複次數低於守則 3 的 20 次，數字只能用來確認流程能跑，不能寫進報告
QUICK = dict(reps=5, matmul_sizes=(512, 1024), infer_batches=(1, 32),
             train_batches=(16, 32), workers=(0, 4), loader_windows=3, loader_window_steps=5)

# 與 train_nima.py 的預設值相同，loader 量的才是「真的訓練」會遇到的速度
TRAIN_BATCH = 32


# ==========================================
# 計時工具
# ==========================================
def timed(fn, warmup, reps, cuda):
    """測速三守則的實作。回傳中位數、IQR、平均與每一次的量測值（毫秒）。"""
    for _ in range(warmup):          # 守則 1
        fn()
    samples = []
    for _ in range(reps):            # 守則 3
        if cuda:
            torch.cuda.synchronize()  # 守則 2：計時前先等 GPU 把之前的工作做完
        t0 = time.perf_counter()
        fn()
        if cuda:
            torch.cuda.synchronize()  # 守則 2：等這一次真的算完才停錶
        samples.append((time.perf_counter() - t0) * 1000)
    return summarize(samples)


def summarize(samples):
    if len(samples) >= 2:
        q1, _, q3 = statistics.quantiles(samples, n=4)
    else:
        q1 = q3 = samples[0]
    return {
        'median_ms': statistics.median(samples),
        'iqr_ms': q3 - q1,
        # 平均值只拿來換算總時間（例如一個 epoch）；偶發的卡頓會拉高平均，
        # 而總時間本來就包含那些卡頓。比較快慢一律看中位數。
        'mean_ms': statistics.fmean(samples),
        'n': len(samples),
        'samples_ms': [round(s, 3) for s in samples],
    }


def fmt(stat):
    return f"{stat['median_ms']:8.1f} ms (IQR {stat['iqr_ms']:5.1f})"


def speedup_text(x):
    return f'{x:.1f}x' if x < 10 else f'{x:.0f}x'


def probe_gpu():
    """
    回傳 (能不能用, 原因)。與 ai_inference.py、check_env.py 相同的兩層檢查：
    is_available() 只代表機器上有裝置；要真的跑一次運算，才知道這版 PyTorch
    支不支援這張卡的架構（例如 RTX 50 系列的 sm_120）。
    """
    if not torch.cuda.is_available():
        return False, '未偵測到 CUDA 裝置'
    try:
        torch.zeros(1, device='cuda').add_(1)
        torch.cuda.synchronize()
        return True, ''
    except Exception as e:
        return False, f'{type(e).__name__}: {e}'


def cap_gpu_memory(margin=0.9):
    """
    把 PyTorch 能配置的顯存上限設為「現在可用顯存」的 90%，超過就拋 OutOfMemoryError。

    Windows 的 NVIDIA 驅動在顯存不夠時，預設不會報 OOM，而是把超出的部分放到系統記憶體
    （sysmem fallback）。結果不是失敗，而是慢上數十倍、看起來像當掉。
    2026-09-22 第一次完整量測就卡在 FP32 batch 256：當時另有其他程式也在用 GPU，
    顯存 100% 滿、好幾分鐘沒有進度。就算沒有其他程式也放不下——batch 128 的峰值是
    9.5 GB，256 推算約需 19 GB，超過 16 GB 的卡。
    量測要的是「放不下就明確失敗」，而且以「可用」而非「總量」為準：
    桌面與其他程式本來就佔著一部分顯存。
    """
    free, total = torch.cuda.mem_get_info()
    torch.cuda.set_per_process_memory_fraction(free * margin / total)
    return free, total


def cpu_name():
    """CPU 的完整型號。platform.processor() 在 Windows 只給得出 'Intel64 Family 6 ...'。"""
    if sys.platform == 'win32':
        try:
            import winreg
            key = r'HARDWARE\DESCRIPTION\System\CentralProcessor\0'
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key) as k:
                return winreg.QueryValueEx(k, 'ProcessorNameString')[0].strip()
        except OSError:
            pass
    cpuinfo = Path('/proc/cpuinfo')
    if cpuinfo.exists():
        for line in cpuinfo.read_text(errors='replace').splitlines():
            if line.startswith('model name'):
                return line.split(':', 1)[1].strip()
    return platform.processor() or '未知'


def free_gpu():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ==========================================
# 各量測項目
# ==========================================
def section_env(gpu_ok, gpu_reason):
    info = {
        'time': datetime.now().isoformat(timespec='seconds'),
        'python': platform.python_version(),
        'os': platform.platform(),
        'torch': torch.__version__,
        'cuda': torch.version.cuda,
        'cudnn': torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        'cpu': cpu_name(),
        'cpu_logical_cores': os.cpu_count(),
        'torch_cpu_threads': torch.get_num_threads(),
        'gpu_usable': gpu_ok,
    }
    if gpu_ok:
        prop = torch.cuda.get_device_properties(0)
        info.update(gpu=prop.name,
                    vram_gb=round(prop.total_memory / 1024 ** 3, 1),
                    capability=f'{prop.major}.{prop.minor}',
                    bf16_supported=torch.cuda.is_bf16_supported())
    else:
        info['gpu_reason'] = gpu_reason

    print('\n=== 1. 環境資訊 ===')
    print(f"  GPU        {info.get('gpu', '無法使用：' + gpu_reason)}")
    if gpu_ok:
        print(f"  顯存       {info['vram_gb']} GB   運算能力 {info['capability']}   "
              f"BF16 {'支援' if info['bf16_supported'] else '不支援'}")
    print(f"  CPU        {info['cpu']}（{info['cpu_logical_cores']} 執行緒，"
          f"PyTorch 使用 {info['torch_cpu_threads']}）")
    print(f"  版本       Python {info['python']} / PyTorch {info['torch']} / "
          f"CUDA {info['cuda']} / cuDNN {info['cudnn']}")
    return info


def section_matmul(cfg, gpu_ok):
    print('\n=== 2. 矩陣乘法 CPU vs GPU（FP32）===')
    rows = []
    for n in cfg['matmul_sizes']:
        a, b = torch.randn(n, n), torch.randn(n, n)
        cpu = timed(lambda: a @ b, warmup=3, reps=cfg['reps'], cuda=False)
        row = {'n': n, 'cpu': cpu, 'cpu_gflops': 2 * n ** 3 / (cpu['median_ms'] / 1000) / 1e9}
        line = f'  n={n:5d}  CPU {fmt(cpu)}'
        if gpu_ok:
            ag, bg = a.cuda(), b.cuda()
            gpu = timed(lambda: ag @ bg, warmup=10, reps=cfg['reps'], cuda=True)
            row.update(gpu=gpu, speedup=cpu['median_ms'] / gpu['median_ms'],
                       gpu_tflops=2 * n ** 3 / (gpu['median_ms'] / 1000) / 1e12)
            line += (f'  | GPU {fmt(gpu)}  | 快 {speedup_text(row["speedup"]):>6}'
                     f'  | {row["gpu_tflops"]:5.2f} TFLOPS')
            del ag, bg
        rows.append(row)
        print(line)
    free_gpu()
    return rows


def section_infer(cfg, gpu_ok):
    """
    兩個模型各跑一次 = 評一張照片的推論成本。
    兩個模型架構相同（只差分類頭），權重不影響速度，所以直接用未載入權重的模型量，
    這一項在沒有 .pth 的機器上也能跑。
    """
    from common import NIMABaseline

    print('\n=== 3. 推論：美感 + 技術兩個模型各跑一次 ===')
    aes = NIMABaseline('distribution', pretrained=False).eval()
    tech = NIMABaseline('single', pretrained=False).eval()

    def run_both(x):
        with torch.no_grad():
            aes(x)
            tech(x)

    x = torch.randn(1, 3, 224, 224)
    res = {'cpu_batch1': timed(lambda: run_both(x), warmup=5, reps=cfg['reps'], cuda=False)}
    print(f"  CPU  batch 1  {fmt(res['cpu_batch1'])}")

    if gpu_ok:
        aes.cuda()
        tech.cuda()
        xg = x.cuda()
        # 模型剛搬上 GPU 後的第一次呼叫。前台評第一張照片時會遇到這個延遲，
        # 這也正是守則 1 要把它丟掉的原因——它不代表穩定狀態的速度。
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        run_both(xg)
        torch.cuda.synchronize()
        res['gpu_first_call_ms'] = (time.perf_counter() - t0) * 1000

        res['gpu_batch1'] = timed(lambda: run_both(xg), warmup=10, reps=cfg['reps'], cuda=True)
        res['speedup'] = res['cpu_batch1']['median_ms'] / res['gpu_batch1']['median_ms']
        print(f"  GPU  batch 1  {fmt(res['gpu_batch1'])}   快 {speedup_text(res['speedup'])}"
              f"   （第一次呼叫 {res['gpu_first_call_ms']:.1f} ms）")

        res['gpu_batches'] = []
        for bs in cfg['infer_batches']:
            xb = torch.randn(bs, 3, 224, 224, device='cuda')
            free_gpu()
            torch.cuda.reset_peak_memory_stats()
            t = timed(lambda: run_both(xb), warmup=5, reps=cfg['reps'], cuda=True)
            row = {'batch': bs, 'step': t,
                   'images_per_s': bs / (t['median_ms'] / 1000),
                   'peak_mb': torch.cuda.max_memory_allocated() / MB}
            res['gpu_batches'].append(row)
            print(f"  GPU  batch {bs:<3d}{fmt(t)}   {row['images_per_s']:7.0f} 張/秒"
                  f"   峰值顯存 {row['peak_mb']:6.0f} MB")
            del xb
        del xg
    del aes, tech
    free_gpu()
    return res


def section_pipeline(cfg, photos_arg):
    """
    單張照片從檔案到評分結果的時間拆解，量的是前台實際呼叫 evaluate_photo() 的那條路。
    這裡刻意呼叫 ai_inference 的內部函式（_load_image_array、_analyze_technical_issues），
    因為要量的正是 evaluate_photo 內部用的那幾段程式，不另外寫一份。
    """
    print('\n=== 4. 單張照片全流程（evaluate_photo）===')
    from PIL import Image
    try:
        import ai_inference as ai
    except Exception as e:  # 缺權重時 ai_inference 會拋 FileNotFoundError
        print(f'  [WARN] 無法載入 ai_inference，略過：{type(e).__name__}: {e}')
        return {'skipped': f'{type(e).__name__}: {e}'}

    if photos_arg:
        photos = [Path(p) for p in photos_arg]
    else:
        # 預設挑 data/my_photos 裡最大的 JPG（最接近相機原檔）與第一個 RAW
        photo_dir = BASE_DIR / 'data' / 'my_photos'
        files = sorted(p for p in photo_dir.iterdir() if p.is_file()) if photo_dir.is_dir() else []
        jpgs = [p for p in files if p.suffix.lower() in ('.jpg', '.jpeg')]
        raws = [p for p in files if p.suffix.lower() in ai.RAW_EXTENSIONS]
        photos = ([max(jpgs, key=lambda p: p.stat().st_size)] if jpgs else []) + raws[:1]

    if not photos:
        print('  [WARN] 沒有可量的照片，略過。可用 --photo 指定，或放照片到 data/my_photos')
        return {'skipped': '沒有照片'}

    cuda = ai.DEVICE.type == 'cuda'
    rows = []
    for path in photos:
        p = str(path)
        try:
            image = ai._load_image_array(p)
        except Exception as e:
            print(f'  [WARN] {path.name} 讀取失敗，略過：{type(e).__name__}: {e}')
            continue
        if ai.evaluate_photo(p, image=image) is None:
            print(f'  [WARN] {path.name} 評分失敗，略過：{ai.LAST_ERROR}')
            continue

        decode = timed(lambda: ai._load_image_array(p), warmup=1, reps=cfg['reps'], cuda=False)
        # 前處理：把原尺寸影像縮到 256 再裁成 224。相機原檔動輒 2,400 萬畫素，這一步不便宜
        preprocess = timed(lambda: ai.TRANSFORM(Image.fromarray(image)), warmup=2,
                           reps=cfg['reps'], cuda=False)
        analysis = timed(lambda: ai._analyze_technical_issues(image), warmup=2,
                         reps=cfg['reps'], cuda=False)
        # 傳入 image= 會跳過解碼，其餘步驟與前台呼叫時完全相同
        rest = timed(lambda: ai.evaluate_photo(p, image=image), warmup=3,
                     reps=cfg['reps'], cuda=cuda)
        # 推論 = 整段扣掉另外量到的兩段。各段中位數相減只是估計，下限取 0
        infer_ms = max(rest['median_ms'] - preprocess['median_ms'] - analysis['median_ms'], 0.0)
        total = decode['median_ms'] + rest['median_ms']
        row = {'photo': path.name, 'width': int(image.shape[1]), 'height': int(image.shape[0]),
               'device': str(ai.DEVICE), 'decode': decode, 'preprocess': preprocess,
               'analysis': analysis, 'evaluate_with_image': rest, 'inference_ms_est': infer_ms,
               'total_ms_est': total, 'decode_share': decode['median_ms'] / total}
        rows.append(row)
        print(f"  {path.name}（{row['width']}x{row['height']}，{row['device']}）")
        print(f"    解碼          {fmt(decode)}")
        print(f"    前處理        {fmt(preprocess)}")
        print(f"    推論          {infer_ms:8.1f} ms（估計：evaluate_photo 扣掉前處理與影像量測）")
        print(f"    影像量測      {fmt(analysis)}")
        print(f"    合計約 {total:.0f} ms，解碼佔 {row['decode_share']:.0%}")
    return {'rows': rows}


def _train_step(model, opt, x, y, dtype, scaler):
    """一個完整的訓練步：前向、EMD loss、反向、更新。dtype=None 代表 FP32。"""
    from common import emd_loss

    def step():
        opt.zero_grad(set_to_none=True)
        if dtype is None:
            loss = emd_loss(model(x), y)
        else:
            with torch.autocast('cuda', dtype=dtype):
                out = model(x)
            loss = emd_loss(out.float(), y)
        if scaler is None:
            loss.backward()
            opt.step()
        else:
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
    return step


def section_train(cfg, gpu_ok):
    print('\n=== 5. 訓練單步：batch size x 精度（資料預先放在 GPU 上，只量運算）===')
    if not gpu_ok:
        print('  [WARN] 沒有可用的 GPU，略過')
        return {'skipped': '沒有可用的 GPU'}
    from common import NIMABaseline

    # BF16 的動態範圍與 FP32 相同，不需要 GradScaler，但要 Ampere（運算能力 8.0）以上。
    # 不支援時退回 FP16 + GradScaler（投影片 4-6）。
    if torch.cuda.is_bf16_supported():
        amp_name, amp_dtype, scaler = 'BF16', torch.bfloat16, None
    else:
        amp_name, amp_dtype, scaler = 'FP16', torch.float16, torch.amp.GradScaler('cuda')

    model = NIMABaseline('distribution', pretrained=False).cuda().train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-5)
    rows = []
    for name, dtype in (('FP32', None), (amp_name, amp_dtype)):
        for bs in cfg['train_batches']:
            free_gpu()
            torch.cuda.reset_peak_memory_stats()
            try:
                x = torch.randn(bs, 3, 224, 224, device='cuda')
                y = torch.softmax(torch.randn(bs, 10, device='cuda'), dim=1)
                step = _train_step(model, opt, x, y, dtype, scaler if dtype is not None else None)
                t = timed(step, warmup=5, reps=cfg['reps'], cuda=True)
            except torch.cuda.OutOfMemoryError:
                rows.append({'precision': name, 'batch': bs, 'oom': True})
                print(f'  {name:5s} batch {bs:<4d} 顯存不足（OOM），更大的 batch 不再嘗試')
                x = y = step = None
                opt.zero_grad(set_to_none=True)
                free_gpu()
                break
            row = {'precision': name, 'batch': bs, 'step': t,
                   'images_per_s': bs / (t['median_ms'] / 1000),
                   'peak_mb': torch.cuda.max_memory_allocated() / MB}
            rows.append(row)
            print(f"  {name:5s} batch {bs:<4d}{fmt(t)}   {row['images_per_s']:7.0f} 張/秒"
                  f"   峰值顯存 {row['peak_mb'] / 1024:5.2f} GB")
            del x, y, step
    del model, opt
    free_gpu()
    return {'amp': amp_name, 'rows': rows,
            'note': '峰值顯存包含模型參數、梯度與 AdamW 狀態；混合精度只做量測，訓練腳本未採用'}


def section_loader(cfg, gpu_ok):
    print(f'\n=== 6. 資料管線：num_workers 對實際訓練速度的影響（batch {TRAIN_BATCH}）===')
    csv_path = BASE_DIR / 'data' / 'ava_train.csv'
    img_dir = BASE_DIR / 'data' / 'dataset'
    if not gpu_ok:
        print('  [WARN] 沒有可用的 GPU，略過')
        return {'skipped': '沒有可用的 GPU'}
    if not csv_path.exists() or not img_dir.is_dir():
        print(f'  [WARN] 找不到 {csv_path.relative_to(BASE_DIR)} 或 {img_dir.relative_to(BASE_DIR)}，略過')
        return {'skipped': '缺少訓練資料'}

    from torch.utils.data import DataLoader
    from common import NIMABaseline, build_transform
    from train_nima import AVADataset

    # 與 train_nima.py --mode distribution 相同的資料集與前處理
    ds = AVADataset(str(csv_path), str(img_dir), build_transform('train_aesthetic'),
                    mode='distribution')

    # 先把所有影像讀一遍，讓作業系統的檔案快取就緒。
    # 不做的話，第一個量的設定會吃到冷硬碟、後面的吃到快取，比較就不公平。
    t0 = time.perf_counter()
    for name in ds.filenames:
        with open(img_dir / name, 'rb') as f:
            f.read()
    cache_s = time.perf_counter() - t0
    print(f'  （先把 {len(ds):,} 張讀進系統快取：{cache_s:.1f} 秒）')

    model = NIMABaseline('distribution', pretrained=False).cuda().train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-5)

    # GPU 純運算的時間：資料已經在 GPU 上。這是資料管線再快也快不過的下限。
    xg = torch.randn(TRAIN_BATCH, 3, 224, 224, device='cuda')
    yg = torch.softmax(torch.randn(TRAIN_BATCH, 10, device='cuda'), dim=1)
    floor = timed(_train_step(model, opt, xg, yg, None, None), warmup=5,
                  reps=cfg['reps'], cuda=True)
    del xg, yg
    print(f'  GPU 純運算下限      {fmt(floor)}')

    rows = []
    for nw in [w for w in cfg['workers'] if w <= (os.cpu_count() or 1)]:
        kw = dict(batch_size=TRAIN_BATCH, shuffle=True, num_workers=nw, pin_memory=True,
                  drop_last=True,
                  # 固定打亂順序，每個設定讀到的是同一批照片
                  generator=torch.Generator().manual_seed(0))
        if nw > 0:
            kw.update(persistent_workers=True, prefetch_factor=4)
        loader = DataLoader(ds, **kw)
        it = iter(loader)

        def real_step():
            x, y = next(it)
            x = x.cuda(non_blocking=True)
            y = y.cuda(non_blocking=True)
            _train_step(model, opt, x, y, None, None)()

        # 前 10 批當 warm-up（含 worker 啟動），另外記下它花了多久
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(10):
            real_step()
        torch.cuda.synchronize()
        startup_s = time.perf_counter() - t0

        # 這裡不逐批計時。開了 worker 之後每批時間是「忽快忽慢」的：
        # 批次已經預先備好就很快，沒備好就要等。逐批取中位數會挑到快的那一群，
        # 高估吞吐（實測 num_workers=4 的逐批中位數比 8 快，換算 epoch 卻比 8 慢）。
        # 改成每 N 批為一組量「總時間 / 批數」，重複數組後取中位數與 IQR——
        # 一樣是守則 3 的重複量測，只是每一次量的單位是一組批次。
        steps = cfg['loader_window_steps']
        windows = []
        for _ in range(cfg['loader_windows']):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(steps):
                real_step()
            torch.cuda.synchronize()
            windows.append((time.perf_counter() - t0) * 1000 / steps)
        stat = summarize(windows)
        epoch_s = stat['median_ms'] * len(loader) / 1000
        rows.append({'num_workers': nw, 'step': stat, 'warmup_10_batches_s': startup_s,
                     'batches_per_epoch': len(loader), 'epoch_s_est': epoch_s,
                     'gpu_busy_ratio': floor['median_ms'] / stat['median_ms']})
        print(f'  num_workers={nw:<2d}      {fmt(stat)}   一個 epoch 約 {epoch_s:5.1f} 秒'
              f'   GPU 忙碌約 {rows[-1]["gpu_busy_ratio"]:.0%}')
        del it, loader

    del model, opt
    free_gpu()
    return {'batch': TRAIN_BATCH, 'dataset': str(csv_path.relative_to(BASE_DIR)),
            'cache_warm_s': cache_s, 'gpu_floor': floor, 'rows': rows}


# ==========================================
# 圖表
# ==========================================
def make_charts(res, out):
    try:
        import matplotlib.pyplot as plt
        from common import plotting as P
    except ImportError as e:
        print(f'[WARN] 沒有 matplotlib，略過圖表：{e}（pip install matplotlib）')
        return []
    if P.setup() is None:
        print('[WARN] 找不到中文字型，圖表裡的中文可能會顯示成方框')

    env = res['env']
    hw = f"{env.get('gpu', 'CPU')} / PyTorch {env['torch']}"
    reps = res['config']['reps']
    made = []

    mm = [r for r in res.get('matmul') or [] if 'speedup' in r]
    if mm:
        fig, ax = plt.subplots(figsize=(7, 4))
        bars = ax.bar([f"{r['n']}" for r in mm], [r['speedup'] for r in mm], width=0.5, color=P.BLUE)
        P.label_bars(ax, bars, [speedup_text(r['speedup']) for r in mm])
        ax.set_xlabel('矩陣大小 n（n x n 乘 n x n）')
        ax.set_ylabel('GPU 比 CPU 快幾倍')
        ax.margins(y=0.15)
        P.title(ax, '矩陣乘法：GPU 對 CPU 的加速倍數',
                f'{hw}，FP32，各 {reps} 次取中位數')
        P.save(fig, out / 'matmul.png')
        made.append('matmul.png')

    inf = res.get('infer') or {}
    if 'gpu_batch1' in inf:
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4), gridspec_kw={'wspace': 0.35})
        vals = [inf['cpu_batch1']['median_ms'], inf['gpu_batch1']['median_ms']]
        bars = a1.barh(['CPU', 'GPU'], vals, height=0.32, color=P.BLUE)
        P.label_bars(a1, bars, [f'{v:.1f} ms' for v in vals], horizontal=True)
        P.category_axis(a1, 'y')
        a1.invert_yaxis()
        a1.grid(axis='y', visible=False)
        a1.grid(axis='x', visible=True)
        a1.set_xlabel('每張照片（毫秒，越短越好）')
        a1.margins(x=0.2)
        P.title(a1, '推論延遲：batch 1',
                f"兩個模型各跑一次，GPU 快 {speedup_text(inf['speedup'])}")

        rows = inf['gpu_batches']
        xs = list(range(len(rows)))
        a2.plot(xs, [r['images_per_s'] for r in rows], marker='o', color=P.BLUE)
        a2.set_xticks(xs, [str(r['batch']) for r in rows])
        a2.set_xlabel('batch size')
        a2.set_ylabel('張 / 秒')
        a2.set_ylim(bottom=0)
        last = rows[-1]
        a2.annotate(f"{last['images_per_s']:,.0f} 張/秒", (xs[-1], last['images_per_s']),
                    xytext=(-6, 8), textcoords='offset points', ha='right',
                    fontsize=9, fontweight='bold', color=P.INK)
        P.title(a2, 'GPU 批次推論吞吐', '一次送多張照片進模型')
        P.save(fig, out / 'inference.png')
        made.append('inference.png')

    pipe = (res.get('pipeline') or {}).get('rows') or []
    if pipe:
        fig, ax = plt.subplots(figsize=(10, 1.2 + 0.8 * len(pipe)))
        names = [r['photo'] for r in pipe]
        # 堆疊長條只有相鄰色塊需要分得開，四色依固定順序使用
        parts = [('解碼', [r['decode']['median_ms'] for r in pipe], P.BLUE),
                 ('前處理', [r['preprocess']['median_ms'] for r in pipe], P.ORANGE),
                 ('推論', [r['inference_ms_est'] for r in pipe], P.AQUA),
                 ('影像量測', [r['analysis']['median_ms'] for r in pipe], P.YELLOW)]
        left = [0.0] * len(pipe)
        for label, vals, color in parts:
            ax.barh(names, vals, left=left, height=0.45, color=color, label=label,
                    edgecolor=P.SURFACE, linewidth=2)
            left = [l + v for l, v in zip(left, vals)]
        for i, r in enumerate(pipe):
            ax.annotate(f"共 {r['total_ms_est']:,.0f} ms，解碼佔 {r['decode_share']:.0%}",
                        (left[i], i), xytext=(6, 0), textcoords='offset points',
                        va='center', fontsize=9, fontweight='bold', color=P.INK)
        P.category_axis(ax, 'y')
        ax.invert_yaxis()
        ax.grid(axis='y', visible=False)
        ax.grid(axis='x', visible=True)
        ax.set_xlabel('毫秒')
        ax.margins(x=0.35)
        ax.legend(loc='lower right', ncol=4, bbox_to_anchor=(1, 1.0))
        P.title(ax, '單張照片全流程的時間拆解', f"evaluate_photo()，裝置 {pipe[0]['device']}")
        P.save(fig, out / 'pipeline.png')
        made.append('pipeline.png')

    tr = [r for r in (res.get('train') or {}).get('rows') or [] if not r.get('oom')]
    if tr:
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4), gridspec_kw={'wspace': 0.3})
        batches = sorted({r['batch'] for r in tr})
        pos = {b: i for i, b in enumerate(batches)}
        precisions = []
        for r in tr:
            if r['precision'] not in precisions:
                precisions.append(r['precision'])
        for prec, color in zip(precisions, P.SERIES):
            pts = [r for r in tr if r['precision'] == prec]
            xs = [pos[r['batch']] for r in pts]
            a1.plot(xs, [r['images_per_s'] for r in pts], marker='o', color=color, label=prec)
            a2.plot(xs, [r['peak_mb'] / 1024 for r in pts], marker='o', color=color, label=prec)
        for ax in (a1, a2):
            ax.set_xticks(range(len(batches)), [str(b) for b in batches])
            ax.set_xlabel('batch size')
            ax.set_ylim(bottom=0)
        a1.set_ylabel('張 / 秒')
        a2.set_ylabel('GB')
        a1.legend(loc='upper left')
        oom = [r for r in res['train']['rows'] if r.get('oom')]
        sub = '資料預先放在 GPU 上，只量運算'
        if oom:
            sub += '；' + '、'.join(f"{r['precision']} batch {r['batch']} OOM" for r in oom)
        P.title(a1, '訓練吞吐', sub)
        P.title(a2, '訓練峰值顯存', f"含模型、梯度與 AdamW 狀態；量測上限 {env.get('memory_cap_gb', '?')} GB")
        P.save(fig, out / 'train_batch.png')
        made.append('train_batch.png')

    ld = res.get('loader') or {}
    if ld.get('rows'):
        rows = ld['rows']
        fig, ax = plt.subplots(figsize=(8, 4.2))
        vals = [r['step']['median_ms'] for r in rows]
        bars = ax.bar([str(r['num_workers']) for r in rows], vals, width=0.5, color=P.BLUE)
        P.label_bars(ax, bars, [f'{v:.0f} ms' for v in vals])
        floor = ld['gpu_floor']['median_ms']
        # 虛線穿過長條，圖內找不到不壓到長條的位置放說明，所以寫在副標
        ax.axhline(floor, color=P.INK_2, linewidth=1, linestyle=(0, (4, 3)))
        ax.set_xlabel('DataLoader num_workers')
        ax.set_ylabel('每批毫秒（越短越好）')
        ax.margins(y=0.15)
        first, best = rows[0], min(rows, key=lambda r: r['step']['median_ms'])
        P.title(ax, '資料管線：num_workers 對訓練速度的影響',
                f"美感模型、batch {ld['batch']}；一個 epoch 約 {first['epoch_s_est']:.0f} 秒"
                f"（{first['num_workers']}）→ {best['epoch_s_est']:.0f} 秒（{best['num_workers']}）；"
                f"虛線 = GPU 純運算下限 {floor:.0f} ms")
        P.save(fig, out / 'loader.png')
        made.append('loader.png')

    return made


# ==========================================
# 表格摘要
# ==========================================
def write_summary(res, out, charts):
    env = res['env']
    L = [f"# GPU / CPU 效能量測（{env['time']}）", '']
    if res['config']['quick']:
        L += ['> **快速模式**：重複次數少於 20 次，數字只能用來確認流程，不可寫進報告。', '']
    L += ['方法：投影片 4-3 測速三守則——warm-up、計時前後 synchronize、'
          f"重複 {res['config']['reps']} 次取中位數與 IQR。", '',
          '## 環境', '', '| 項目 | 內容 |', '|---|---|',
          f"| GPU | {env.get('gpu', '無法使用')} |"]
    if env.get('gpu'):
        L += [f"| 顯存 / 運算能力 | {env['vram_gb']} GB / {env['capability']} |",
              f"| 量測時的顯存上限 | {env.get('memory_cap_gb', '?')} GB（啟動時可用顯存的 90%，超過視為 OOM） |"]
    L += [f"| CPU | {env['cpu']}（{env['cpu_logical_cores']} 執行緒） |",
          f"| 版本 | Python {env['python']}、PyTorch {env['torch']}、CUDA {env['cuda']}、cuDNN {env['cudnn']} |",
          f"| 作業系統 | {env['os']} |", '']

    if res.get('matmul'):
        L += ['## 矩陣乘法（FP32）', '', '| n | CPU 中位數 (IQR) | GPU 中位數 (IQR) | 加速 | GPU TFLOPS |',
              '|---|---|---|---|---|']
        for r in res['matmul']:
            g = r.get('gpu')
            L.append(f"| {r['n']} | {r['cpu']['median_ms']:.2f} ms ({r['cpu']['iqr_ms']:.2f}) | "
                     + (f"{g['median_ms']:.2f} ms ({g['iqr_ms']:.2f}) | {speedup_text(r['speedup'])} | "
                        f"{r['gpu_tflops']:.2f} |" if g else '- | - | - |'))
        L.append('')

    inf = res.get('infer')
    if inf:
        L += ['## 推論（美感 + 技術兩個模型各跑一次）', '', '| 裝置 | batch | 中位數 (IQR) | 張/秒 | 峰值顯存 |',
              '|---|---|---|---|---|',
              f"| CPU | 1 | {inf['cpu_batch1']['median_ms']:.1f} ms ({inf['cpu_batch1']['iqr_ms']:.1f}) | - | - |"]
        for r in inf.get('gpu_batches', []):
            L.append(f"| GPU | {r['batch']} | {r['step']['median_ms']:.1f} ms ({r['step']['iqr_ms']:.1f}) | "
                     f"{r['images_per_s']:,.0f} | {r['peak_mb']:.0f} MB |")
        if 'gpu_first_call_ms' in inf:
            L += ['', f"GPU 第一次呼叫：{inf['gpu_first_call_ms']:.1f} ms（不計入上表）。"]
        L.append('')

    pipe = (res.get('pipeline') or {}).get('rows')
    if pipe:
        L += ['## 單張照片全流程（evaluate_photo）', '',
              '| 照片 | 尺寸 | 解碼 | 前處理 | 推論（估計） | 影像量測 | 合計 | 解碼佔比 |',
              '|---|---|---|---|---|---|---|---|']
        for r in pipe:
            L.append(f"| {r['photo']} | {r['width']}x{r['height']} | {r['decode']['median_ms']:.1f} ms | "
                     f"{r['preprocess']['median_ms']:.1f} ms | {r['inference_ms_est']:.1f} ms | "
                     f"{r['analysis']['median_ms']:.1f} ms | {r['total_ms_est']:.0f} ms | {r['decode_share']:.0%} |")
        L += ['', f"裝置：{pipe[0]['device']}。推論 = evaluate_photo(image=...) 的中位數扣掉前處理與影像量測，"
                  '是估計值；精確的推論時間看上面「推論」一節。', '']

    tr = res.get('train') or {}
    if tr.get('rows'):
        L += ['## 訓練單步（資料預先放在 GPU 上）', '',
              '| 精度 | batch | 每步中位數 (IQR) | 張/秒 | 峰值顯存 |', '|---|---|---|---|---|']
        for r in tr['rows']:
            if r.get('oom'):
                L.append(f"| {r['precision']} | {r['batch']} | OOM | - | - |")
            else:
                L.append(f"| {r['precision']} | {r['batch']} | {r['step']['median_ms']:.1f} ms "
                         f"({r['step']['iqr_ms']:.1f}) | {r['images_per_s']:,.0f} | {r['peak_mb'] / 1024:.2f} GB |")
        L += ['', tr['note'] + '。', '']

    ld = res.get('loader') or {}
    if ld.get('rows'):
        L += [f"## 資料管線（{ld['dataset']}，batch {ld['batch']}）", '',
              f"GPU 純運算下限：{ld['gpu_floor']['median_ms']:.1f} ms（IQR {ld['gpu_floor']['iqr_ms']:.1f}）", '',
              '| num_workers | 每批（中位數，IQR） | 一個 epoch 約 | GPU 忙碌 |', '|---|---|---|---|']
        for r in ld['rows']:
            L.append(f"| {r['num_workers']} | {r['step']['median_ms']:.1f} ms ({r['step']['iqr_ms']:.1f}) | "
                     f"{r['epoch_s_est']:.1f} 秒 | {r['gpu_busy_ratio']:.0%} |")
        cfg = res['config']
        L += ['', f"每 {cfg['loader_window_steps']} 批為一組量「總時間 / 批數」，共 {cfg['loader_windows']} 組，"
                  '取中位數與 IQR。開了 worker 之後每批時間忽快忽慢，逐批計時會高估吞吐。'
                  f"量測前先把影像全部讀過一次（{ld['cache_warm_s']:.1f} 秒），排除冷硬碟的影響。", '']

    skipped = [f"- {k}：{v['skipped']}" for k, v in res.items()
               if isinstance(v, dict) and 'skipped' in v]
    if skipped:
        L += ['## 略過的項目', ''] + skipped + ['']
    if charts:
        L += ['## 圖表', ''] + [f'![{c}]({c})' for c in charts] + ['']
    (out / 'summary.md').write_text('\n'.join(L), encoding='utf-8')


# ==========================================
# 主程式
# ==========================================
def parse_args():
    p = argparse.ArgumentParser(description='GPU / CPU 效能量測（照投影片 4-3 測速三守則）')
    p.add_argument('--quick', action='store_true',
                   help='快速模式：減少重複次數與量測點，只用來確認流程，數字不可寫進報告')
    p.add_argument('--only', metavar='項目',
                   help=f"只跑指定項目，逗號分隔：{','.join(SECTIONS[1:])}（env 一律執行）")
    p.add_argument('--photo', action='append', metavar='路徑',
                   help='pipeline 要量的照片，可重複指定；預設取 data/my_photos 最大的 JPG 與第一個 RAW')
    p.add_argument('--out', metavar='資料夾', help='輸出資料夾（預設 reports/benchmark/<時間>）')
    p.add_argument('--no-charts', action='store_true', help='不產生圖表')
    return p.parse_args()


def main():
    args = parse_args()
    wanted = set(SECTIONS)
    if args.only:
        wanted = {s.strip() for s in args.only.split(',') if s.strip()} | {'env'}
        unknown = wanted - set(SECTIONS)
        if unknown:
            print(f"[FAIL] 不認得的項目：{', '.join(sorted(unknown))}｜可用：{', '.join(SECTIONS[1:])}")
            return 1

    cfg = QUICK if args.quick else FULL
    out = Path(args.out) if args.out else (
        BASE_DIR / 'reports' / 'benchmark' / datetime.now().strftime('%Y%m%d-%H%M%S'))
    out.mkdir(parents=True, exist_ok=True)

    gpu_ok, gpu_reason = probe_gpu()
    res = {'config': {**cfg, 'quick': args.quick, 'sections': sorted(wanted)},
           'env': section_env(gpu_ok, gpu_reason)}
    if gpu_ok:
        free, total = cap_gpu_memory()
        res['env'].update(vram_free_gb_at_start=round(free / 1024 ** 3, 1),
                          memory_cap_gb=round(free * 0.9 / 1024 ** 3, 1))
        print(f"  顯存上限   {res['env']['memory_cap_gb']} GB（啟動時可用 "
              f"{res['env']['vram_free_gb_at_start']} GB 的 90%，超過就視為 OOM）")
    if args.quick:
        print('\n[INFO] 快速模式：重複次數少於 20 次，數字只能用來確認流程，不可寫進報告')

    def save_json():
        # 每做完一項就存一次，中途中斷也保得住已經量到的數字
        with open(out / 'benchmark.json', 'w', encoding='utf-8') as f:
            json.dump(res, f, ensure_ascii=False, indent=2)

    save_json()
    steps = [('matmul', lambda: section_matmul(cfg, gpu_ok)),
             ('infer', lambda: section_infer(cfg, gpu_ok)),
             ('pipeline', lambda: section_pipeline(cfg, args.photo)),
             ('train', lambda: section_train(cfg, gpu_ok)),
             ('loader', lambda: section_loader(cfg, gpu_ok))]
    for name, run in steps:
        if name in wanted:
            res[name] = run()
            save_json()

    charts = [] if args.no_charts else make_charts(res, out)
    write_summary(res, out, charts)
    print(f'\n[ OK ] 已寫入 {out}')
    print(f'       benchmark.json、summary.md' + (f"、{len(charts)} 張圖表" if charts else ''))
    return 0


if __name__ == '__main__':
    sys.exit(main())
