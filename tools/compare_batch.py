"""Compare sequential decoding with the production four-decoder pipeline.

Both use the same model, images, weights, feature extraction and JSONL writes.
Model inference remains serial in BOTH modes. No synthetic scores are generated.
"""
import sys
from pathlib import Path

# 這支程式在子資料夾裡；共用模組（ai_inference、common 等）在專案根目錄
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json
import math
import platform
import statistics
import time

SCORE_KEYS = ('aesthetic_score', 'technical_score', 'overall_score')


def sequential_batch(paths, output, weight, ai):
    success = 0
    with Path(output).open('x', encoding='utf-8') as handle:
        for path in paths:
            path = str(Path(path).resolve())
            try:
                image = ai.load_image(path)
            except Exception as exc:
                record = dict(path=path, ok=False, stage='decode', error=str(exc))
            else:
                try:
                    result = ai.evaluate_photo(path, image=image,
                                               aesthetic_weight=weight,
                                               return_features=True)
                    error = ai.LAST_ERROR if result is None else None
                finally:
                    del image
                record = (dict(path=path, ok=True, **result) if result is not None
                          else dict(path=path, ok=False, stage='inference', error=error))
            if record['ok']:
                success += 1
            else:
                record.update(aesthetic_weight=weight, technical_weight=1-weight)
            handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
            handle.flush()
    return dict(total=len(paths), success=success, failed=len(paths)-success)


def read_records(path):
    with open(path, encoding='utf-8') as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    records = {row['path']: row for row in rows}
    if len(records) != len(rows):
        raise ValueError('Duplicate paths in results')
    return records


def compare_records(left, right, atol=1e-6):
    if not math.isfinite(atol) or atol < 0:
        raise ValueError('atol must be finite and nonnegative')
    if left.keys() != right.keys():
        raise ValueError('The two runs contain different photos')
    differences, failures = [], []
    maximum = {key: 0.0 for key in SCORE_KEYS}
    for path in sorted(left):
        a, b = left[path], right[path]
        if not a['ok'] or not b['ok']:
            failures.append({'path': path, 'sequential_ok': a['ok'], 'parallel_ok': b['ok']})
            continue
        delta = {key: abs(float(a[key])-float(b[key])) for key in SCORE_KEYS}
        if not all(math.isfinite(v) for v in delta.values()):
            raise ValueError('Non-finite score')
        for key, value in delta.items():
            maximum[key] = max(maximum[key], value)
        differences.append({'path': path, 'sequential_scores': {k:a[k] for k in SCORE_KEYS},
                            'parallel_scores': {k:b[k] for k in SCORE_KEYS}, **delta})
    return {'compared': len(differences), 'failed_photos': failures,
            'max_absolute_difference': maximum, 'per_photo_difference': differences,
            'all_scores_match': bool(differences) and not failures and
                                all(value <= atol for value in maximum.values())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder')
    parser.add_argument('--output', required=True, help='New report directory')
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--weight', type=float, default=0.6)
    parser.add_argument('--atol', type=float, default=1e-6)
    args = parser.parse_args()
    if (args.repeats < 2 or not math.isfinite(args.weight) or not 0 <= args.weight <= 1
            or not math.isfinite(args.atol) or args.atol < 0):
        parser.error('repeats >= 2, weight in [0,1], finite atol >= 0 required')
    import ai_inference as ai
    from batch_pipeline import scan_folder, analyze_batch
    paths, skipped = scan_folder(args.folder)
    if not paths:
        parser.error('No supported photos')
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)

    def synchronize():
        if ai.DEVICE.type == 'cuda':
            ai.torch.cuda.synchronize()

    # One complete pass warms models and filesystem cache in each mode.
    # Initialization is excluded; timed runs include decoding and JSONL I/O.
    runners = {
        'sequential': lambda dest: sequential_batch(paths, dest, args.weight, ai),
        'parallel': lambda dest: analyze_batch(paths, dest, aesthetic_weight=args.weight),
    }
    for mode, run in runners.items():
        summary = run(out / f'warmup-{mode}.jsonl')
        if summary['success'] == 0:
            raise RuntimeError(f'{mode}: no successful images; see warmup output')
    times = {mode: [] for mode in runners}
    comparisons = []
    report = {'python': platform.python_version(), 'platform': platform.platform(),
              'torch': ai.torch.__version__, 'device': str(ai.DEVICE),
              'model_version': ai.model_version(), 'weight': args.weight,
              'photo_count': len(paths), 'paths': [str(p) for p in paths],
              'skipped_extensions': skipped, 'repeats': args.repeats, 'atol': args.atol,
              'method': 'warm cache; alternating order; decode + inference + JSONL I/O',
              'seconds': times, 'comparisons': comparisons}
    for repeat in range(args.repeats):
        modes = list(runners) if repeat % 2 == 0 else list(reversed(runners))
        files = {}
        for mode in modes:
            files[mode] = out / f'{repeat+1:02d}-{mode}.jsonl'
            synchronize()
            start = time.perf_counter()
            runners[mode](files[mode])
            synchronize()
            times[mode].append(time.perf_counter()-start)
        comparisons.append(compare_records(read_records(files['sequential']),
                                           read_records(files['parallel']), args.atol))
        (out / 'comparison.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    medians = {mode: statistics.median(values) for mode, values in times.items()}
    report.update(median_seconds=medians,
                  speedup=medians['sequential']/medians['parallel'],
                  all_scores_match=all(r['all_scores_match'] for r in comparisons))
    (out / 'comparison.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    lines = ['# 循序與平行解碼比較', '',
             f"照片：{len(paths)} 張；每種模式 {args.repeats} 次；裝置：{ai.DEVICE}", '',
             '| 模式 | 耗時中位數（秒） | 每秒照片數 |', '|---|---:|---:|']
    for mode in runners:
        lines.append(f'| {mode} | {medians[mode]:.4f} | {len(paths)/medians[mode]:.2f} |')
    lines += ['', f"加速比（循序／平行）：{report['speedup']:.3f}",
              f"全部照片分數一致（絕對誤差 <= {args.atol}）：{report['all_scores_match']}", '',
              '兩種模式都只由一個執行緒執行模型；parallel 使用 4 個解碼執行緒。',
              '先完整暖機，交替測試順序；計時包含解碼、推論與結果寫檔，不含模型載入。',
              '這是暖快取測試；失敗照片與逐張三項分數差異詳見 comparison.json。']
    (out / 'summary.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(out / 'summary.md')
    return 0 if report['all_scores_match'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
