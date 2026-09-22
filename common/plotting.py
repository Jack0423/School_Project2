"""
報表圖表的共用樣式。model_report.py 與 benchmark_gpu.py 使用。

刻意不從 common/__init__.py 匯出：matplotlib 只是產生報表用的選用套件，
而 ai_inference.py 會 import common——若在 __init__ 連帶 import matplotlib，
沒裝 matplotlib 的前台會整個載入失敗，失去 AI 功能。
要用的腳本請自己寫 `from common import plotting`。

樣式原則
--------
  * 白底、粗體標題靠左、格線是極淡的實線，資料本身才是最顯眼的東西。
  * 顏色依「用途」分配：一個系列一律用藍色；兩三個系列依固定順序用
    藍 → 橘 → 青綠（這三色兩兩在色盲模擬下仍分得開）；
    對照組、不是重點的東西用灰色。
  * 文字一律用墨色系，不用資料的顏色——淺色的系列色當文字會看不清楚。
  * 兩個以上的系列一定有圖例；不在每個點上標數字，只標重點。
"""
import matplotlib

matplotlib.use('Agg')  # 只存檔、不開視窗。必須在 import pyplot 之前設定

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

SURFACE = '#fcfcfb'   # 圖表底色
INK = '#0b0b0b'       # 主要文字
INK_2 = '#52514e'     # 次要文字（副標、軸標題）
MUTED = '#898781'     # 刻度與註記
GRID = '#e1e0d9'      # 格線
AXIS = '#c3c2b7'      # 座標軸
DEEMPH = '#c3c2b7'    # 非重點的對照組

BLUE, ORANGE, AQUA = '#2a78d6', '#eb6834', '#1baf7a'
SERIES = (BLUE, ORANGE, AQUA)
# 第四色只用在「只有相鄰色塊要分得開」的堆疊長條；散佈圖、折線圖最多用前三色
YELLOW = '#eda100'

# 單一色相、由淺到深：用在混淆矩陣這種「數量大小」的色塊
SEQUENTIAL = LinearSegmentedColormap.from_list('seq_blue', ['#cde2fb', '#6da7ec', '#256abf', '#104281'])

# 依序嘗試的中文字型。Windows 繁中版內建微軟正黑體。
_CJK_FONTS = ('Microsoft JhengHei', 'Microsoft YaHei', 'PingFang TC',
              'Noto Sans CJK TC', 'Noto Sans TC', 'Heiti TC', 'SimHei')


def setup():
    """
    套用全域樣式。回傳實際使用的中文字型名稱；找不到時回傳 None，
    由呼叫端決定要不要警告（圖表仍會產生，只是中文會變成方框）。
    """
    installed = {f.name for f in font_manager.fontManager.ttflist}
    cjk = next((name for name in _CJK_FONTS if name in installed), None)

    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ([cjk] if cjk else []) + ['DejaVu Sans'],
        # 中文字型多半沒有 U+2212 減號，負數會變成方框
        'axes.unicode_minus': False,

        'figure.facecolor': SURFACE,
        'axes.facecolor': SURFACE,
        'savefig.facecolor': SURFACE,
        'savefig.dpi': 150,

        'axes.edgecolor': AXIS,
        'axes.linewidth': 0.8,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.axisbelow': True,
        'axes.grid': True,
        'axes.grid.axis': 'y',
        'grid.color': GRID,
        'grid.linewidth': 0.8,
        'grid.linestyle': '-',

        'axes.titlelocation': 'left',
        'axes.titleweight': 'bold',
        'axes.titlesize': 13,
        'axes.titlecolor': INK,
        'axes.titlepad': 22,   # 留位置給副標
        'axes.labelcolor': INK_2,
        'axes.labelsize': 10,
        'xtick.color': MUTED,
        'ytick.color': MUTED,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,

        'legend.frameon': False,
        'legend.fontsize': 9,
        'legend.labelcolor': INK_2,

        'lines.linewidth': 2,
        'lines.solid_capstyle': 'round',
        'lines.solid_joinstyle': 'round',
        'lines.markersize': 7,
        'axes.prop_cycle': matplotlib.cycler(color=list(SERIES)),
    })
    return cjk


def title(ax, text, subtitle=None):
    """標題靠左加粗；副標用次要文字色寫在標題下方。"""
    ax.set_title(text)
    if subtitle:
        ax.text(0, 1.02, subtitle, transform=ax.transAxes, ha='left', va='bottom',
                fontsize=9.5, color=INK_2)


def label_bars(ax, bars, texts, horizontal=False, pad=3):
    """在長條的尾端標數值（直條標在頂端、橫條標在右端）。"""
    for bar, text in zip(bars, texts):
        if horizontal:
            x, y = bar.get_x() + bar.get_width(), bar.get_y() + bar.get_height() / 2
            ax.annotate(text, (x, y), xytext=(pad, 0), textcoords='offset points',
                        ha='left', va='center', fontsize=9, fontweight='bold', color=INK)
        else:
            x, y = bar.get_x() + bar.get_width() / 2, bar.get_height()
            ax.annotate(text, (x, y), xytext=(0, pad), textcoords='offset points',
                        ha='center', va='bottom', fontsize=9, fontweight='bold', color=INK)


def category_axis(ax, axis='y'):
    """類別軸（例如橫條圖左側的名稱）用次要文字色，比數值刻度更清楚。"""
    ax.tick_params(axis=axis, labelcolor=INK_2, labelsize=10, length=0)


def note(ax, text, x=0.99, y=0.03, ha='right', va='bottom'):
    """圖內的小字註記（例如資料來源、門檻說明）。"""
    ax.text(x, y, text, transform=ax.transAxes, ha=ha, va=va, fontsize=8.5,
            color=MUTED, style='italic')


def save(fig, path):
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
