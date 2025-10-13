# * 支持中文标签：代码会自动尝试寻找系统中文字体以正确显示中文（若找不到字体，请在系统安装中文字体或把字体文件路径给脚本）。
# * 默认读取 `scores.xlsx`；如果该文件不存在，脚本会生成示例数据并绘图，方便演示。
# * 使用的库：`pandas`, `numpy`, `matplotlib`, `openpyxl`。
# * 输出：会保存 `radar_chart.png`，并在 notebook 中显示图像；同时展示读取到的表格。

from pathlib import Path
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os

# 用于向用户展示 DataFrame（UI friendly）
try:
    from caas_jupyter_tools import display_dataframe_to_user
    _have_display = True
except Exception:
    _have_display = False

# ---------- 用户可修改的变量 ----------
filepath = "scores.xlsx"  # 如果你有自己的文件，把路径改成你的文件名/路径
output_png = "radar_chart.png"
title = "成绩雷达图示例"  # 图标题（中文）
# --------------------------------------

# 尝试找到一个支持中文的字体（系统上常见的中文字体名关键字）
def find_chinese_font():
    # 首先检查本地fonts目录
    local_fonts_dir = Path("fonts")
    if local_fonts_dir.exists():
        font_files = list(local_fonts_dir.glob("*.ttf")) + list(local_fonts_dir.glob("*.otf")) + list(local_fonts_dir.glob("*.ttc"))
        for font_file in font_files:
            try:
                prop = fm.FontProperties(fname=str(font_file))
                return str(font_file)
            except Exception:
                continue

    # 常见中文字体候选列表（扩展版）
    candidates = [
        "SimHei", "SimSun", "Microsoft YaHei", "Microsoft YaHei UI",
        "WenQuanYi", "WenQuanYi Zen Hei", "WenQuanYi Micro Hei",
        "Noto Sans CJK", "Noto Sans SC", "Noto Sans TC", "Noto Sans JP", "Noto Sans KR",
        "PingFang", "STHeiti", "STSong", "STKaiti", "STFangsong",
        "AR PL UMing CN", "AR PL UKai CN", "AR PL ShanHeiSun Uni",
        "Source Han Sans", "Source Han Sans CN", "Source Han Serif",
        "Droid Sans Fallback", "Hiragino Sans", "Hiragino Kaku Gothic",
        "Meiryo", "Yu Gothic", "Malgun Gothic", "Apple SD Gothic Neo",
        "Arial Unicode MS", "Code2000", "DejaVu Sans"
    ]

    # 检查系统字体
    sys_fonts = fm.findSystemFonts(fontpaths=None, fontext="ttf")
    for fpath in sys_fonts:
        try:
            prop = fm.FontProperties(fname=fpath)
            name = prop.get_name()
            filename = os.path.basename(fpath).lower()

            for c in candidates:
                if (c.lower() in name.lower() or
                    c.lower().replace(" ", "") in filename or
                    "noto" in filename and ("cjk" in filename or "sc" in filename or "tc" in filename) or
                    "source" in filename and ("han" in filename or "cjk" in filename) or
                    "wenquanyi" in filename or
                    "ar pl" in filename):
                    return fpath
        except Exception:
            continue

    # 如果还是没找到，尝试使用matplotlib的默认中文字体
    try:
        # 尝试使用matplotlib内置的字体
        from matplotlib import font_manager
        font_list = font_manager.get_font_names()
        chinese_fonts = [f for f in font_list if any(keyword in f.lower() for keyword in
                       ['noto', 'cjk', 'han', 'yahei', 'hei', 'song', 'kai'])]
        if chinese_fonts:
            return chinese_fonts[0]
    except Exception:
        pass

    # 最后的备选方案：创建一个简单字体映射
    return None

# 设置字体回退机制
def setup_font():
    font_path = find_chinese_font()
    if font_path:
        try:
            zh_font = fm.FontProperties(fname=font_path)
            # 测试字体是否可以正常显示中文
            import matplotlib
            matplotlib.use('Agg')  # 使用非交互式后端
            test_fig, test_ax = plt.subplots(figsize=(1, 1))
            test_text = test_ax.text(0.5, 0.5, "测试", fontproperties=zh_font)

            # 检查是否有字体警告
            import warnings
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                test_fig.canvas.draw()
                # 如果有字体相关的警告，说明字体不支持中文
                font_warnings = [warning for warning in w if "Glyph" in str(warning.message)]
                plt.close(test_fig)

                if font_warnings:
                    raise Exception(f"字体不支持中文字符: {font_warnings[0].message}")

            print(f"✅ 使用中文字体: {os.path.basename(font_path) if os.path.isfile(font_path) else font_path}")
            return zh_font
        except Exception as e:
            print(f"⚠️ 字体加载失败: {e}")

    # 如果没有中文字体，使用英文标签
    print("⚠️ 未找到可用的中文字体，将使用英文标签")
    return None

# 设置字体
zh_font = setup_font()

# 读取或创建示例数据
def read_scores(path):
    p = Path(path)
    if not p.exists():
        # 创建一个演示用的 xlsx
        if zh_font:
            demo = pd.DataFrame({
                "科目": ["语文", "数学", "英语", "物理", "化学", "生物"],
                "成绩": [88, 95, 82, 76, 84, 90]
            })
        else:
            demo = pd.DataFrame({
                "Subject": ["Chinese", "Math", "English", "Physics", "Chemistry", "Biology"],
                "Score": [88, 95, 82, 76, 84, 90]
            })
        demo.to_excel(p, index=False)
        df = demo
    else:
        df = pd.read_excel(p, engine="openpyxl")
    return df

df = read_scores(filepath)

# 尝试兼容不同的列名（英文/中文）
def ensure_columns(df):
    cols = df.columns.tolist()
    # 如果已经有科目/成绩，直接返回
    if "科目" in cols and "成绩" in cols:
        return df[["科目", "成绩"]].dropna()
    # 尝试找到可能的科目列（非数字）和成绩列（数字）
    possible_score_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    possible_subject_cols = [c for c in cols if not pd.api.types.is_numeric_dtype(df[c])]
    if possible_subject_cols and possible_score_cols:
        sub_col = possible_subject_cols[0]
        score_col = possible_score_cols[0]
        newdf = df[[sub_col, score_col]].copy()
        newdf.columns = ["科目", "成绩"]
        return newdf.dropna()
    # 否则抛出错误
    raise ValueError("找不到合适的科目和成绩列。请确保文件包含 '科目' 和 '成绩' 两列，或一列文本类科目、一列数字类成绩。")

try:
    plot_df = ensure_columns(df)
except Exception as e:
    raise RuntimeError(f"读取数据失败: {e}")

# 向用户展示表格（如果工具可用）
if _have_display:
    display_dataframe_to_user("读取到的成绩表", plot_df)
else:
    print("读取到的成绩表：\n", plot_df)

# 雷达图绘制
labels = plot_df["科目"].astype(str).tolist()
values = plot_df["成绩"].astype(float).tolist()

# 数量
N = len(labels)
if N < 3:
    raise RuntimeError("雷达图至少需要 3 个维度（科目）。当前科目数: {}".format(N))

# 计算角度（每个维度的角度）
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
# 为了闭合多边形，需把第一个点再加到末尾
angles += angles[:1]
values += values[:1]

# 设置最大半径（让图形有一定余量）
max_val = max(values)
# 将最大值上调到最近的 10 的倍数（或至少比 max 高一点）
rmax = math.ceil(max_val / 10.0) * 10
if rmax == 0:
    rmax = 10

# 创建更美观的极坐标图
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True), facecolor='#f8f9fa')

# 设置图形背景为浅色
fig.patch.set_facecolor('#f8f9fa')
ax.set_facecolor('#ffffff')

# 从正上方开始绘制，并使角度顺时针增加（更符合雷达图习惯）
ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)

# 设置更美观的颜色方案
colors = {
    'primary': '#2E86AB',      # 主色调
    'secondary': '#A23B72',    # 次要色
    'accent': '#F18F01',       # 强调色
    'grid': '#E0E0E0',         # 网格线颜色
    'text': '#333333',         # 文本颜色
    'fill': '#2E86AB20'        # 填充色
}

# 设置角度标签 - 更大的字体和更好的颜色
ax.set_thetagrids(np.degrees(angles[:-1]), labels=labels,
                 fontproperties=zh_font, fontsize=14, color=colors['text'],
                 fontweight='500')

# 设置 radial ticks（刻度）- 更好的样式
tick_step = max(5, rmax // 5)
radial_ticks = list(range(0, rmax + 1, tick_step))
ax.set_rgrids(radial_ticks, labels=[str(t) for t in radial_ticks],
              fontproperties=zh_font, fontsize=12, color=colors['text'], alpha=0.8)

# 限制半径范围
ax.set_ylim(0, rmax)

# 绘制数据线 - 使用渐变色和更粗的线条
line = ax.plot(angles, values, linewidth=3, linestyle='solid',
               marker='o', markersize=8, color=colors['primary'],
               markerfacecolor=colors['accent'], markeredgecolor=colors['primary'],
               markeredgewidth=2, alpha=0.9, label='成绩')

# 创建渐变填充效果
fill = ax.fill(angles, values, alpha=0.3, color=colors['primary'],
               edgecolor=colors['primary'], linewidth=2)

# 在每个点上标注具体数值 - 更美观的样式
for i, (angle, val, label) in enumerate(zip(angles[:-1], values[:-1], labels)):
    # 创建背景框让数值更清晰
    bbox_props = dict(boxstyle="round,pad=0.3", facecolor='white',
                      edgecolor=colors['primary'], alpha=0.9, linewidth=1.5)
    ax.annotate(f"{val:.0f}", xy=(angle, val), xytext=(8, 8),
                textcoords='offset points', fontproperties=zh_font,
                fontsize=11, fontweight='bold', color=colors['primary'],
                bbox=bbox_props)

# 标题（支持中英文回退） - 更醒目的样式
if zh_font:
    chart_title = title
else:
    chart_title = "Student Scores Radar Chart"
ax.set_title(chart_title, fontproperties=zh_font, fontsize=20, pad=30,
            fontweight='bold', color=colors['text'])

# 美化网格线和极轴
ax.grid(True, color=colors['grid'], linestyle='--', linewidth=1, alpha=0.6)
ax.spines['polar'].set_linewidth(2)
ax.spines['polar'].set_color(colors['primary'])
ax.spines['polar'].set_alpha(0.8)

# 设置极坐标轴线的样式
ax.spines['start'].set_color(colors['grid'])
ax.spines['start'].set_linewidth(1)
ax.spines['start'].set_alpha(0.6)

# 添加成绩等级区域显示
avg_score = np.mean(values[:-1])
if avg_score > 0:
    # 添加平均分环形线
    circle_angles = np.linspace(0, 2 * np.pi, 100)
    circle_values = [avg_score] * 100
    ax.plot(circle_angles, circle_values, linestyle=':', linewidth=2,
           color=colors['secondary'], alpha=0.7, label=f'平均分: {avg_score:.1f}')

    # 添加成绩等级区域
    # 优秀区域 (85-100分)
    excellent_zone = 85
    if rmax >= excellent_zone:
        excellent_angles = np.linspace(0, 2 * np.pi, 100)
        excellent_values = [excellent_zone] * 100
        ax.fill_between(excellent_angles, excellent_zone, rmax,
                       color='#4CAF50', alpha=0.1, label='优秀')
        ax.plot(excellent_angles, excellent_values, linestyle='-', linewidth=1,
               color='#4CAF50', alpha=0.3)

    # 良好区域 (70-85分)
    good_zone = 70
    if rmax >= good_zone:
        good_angles = np.linspace(0, 2 * np.pi, 100)
        good_values = [good_zone] * 100
        ax.fill_between(good_angles, good_zone, excellent_zone if rmax >= excellent_zone else rmax,
                       color='#2196F3', alpha=0.08, label='良好')
        ax.plot(good_angles, good_values, linestyle='-', linewidth=1,
               color='#2196F3', alpha=0.3)

    # 及格区域 (60-70分)
    pass_zone = 60
    if rmax >= pass_zone:
        pass_angles = np.linspace(0, 2 * np.pi, 100)
        pass_values = [pass_zone] * 100
        ax.fill_between(pass_angles, pass_zone, good_zone if rmax >= good_zone else rmax,
                       color='#FF9800', alpha=0.06, label='及格')
        ax.plot(pass_angles, pass_values, linestyle='-', linewidth=1,
               color='#FF9800', alpha=0.3)

    # 图例
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1),
             frameon=True, fancybox=True, shadow=True,
             prop=zh_font if zh_font else None, fontsize=10)

# 添加悬停效果的模拟（通过高亮最佳和最差科目）
values_without_last = values[:-1]
max_val_idx = np.argmax(values_without_last)
min_val_idx = np.argmin(values_without_last)

# 高亮最高分科目
if zh_font:
    max_label = f"最高分: {labels[max_val_idx]} ({values_without_last[max_val_idx]:.0f}分)"
    min_label = f"最低分: {labels[min_val_idx]} ({values_without_last[min_val_idx]:.0f}分)"
else:
    max_label = f"Highest: {labels[max_val_idx]} ({values_without_last[max_val_idx]:.0f})"
    min_label = f"Lowest: {labels[min_val_idx]} ({values_without_last[min_val_idx]:.0f})"

# 在图表底部添加统计信息
fig.text(0.5, 0.02, f'{max_label} | {min_label}',
         ha='center', fontsize=12, color=colors['text'],
         bbox=dict(boxstyle='round,pad=0.5', facecolor=colors['fill'],
                  edgecolor=colors['primary'], alpha=0.8))

# 添加分数分布的简要统计
std_score = np.std(values_without_last)
if zh_font:
    stats_text = f'标准差: {std_score:.1f} | 范围: {min(values_without_last):.0f}-{max(values_without_last):.0f}'
else:
    stats_text = f'Std Dev: {std_score:.1f} | Range: {min(values_without_last):.0f}-{max(values_without_last):.0f}'
fig.text(0.5, 0.95, stats_text,
         ha='center', fontsize=10, color=colors['secondary'],
         bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                  edgecolor=colors['secondary'], alpha=0.7))

# 优化布局并保存
plt.tight_layout(pad=3.0)
plt.savefig(output_png, dpi=300, bbox_inches='tight',
           facecolor=fig.get_facecolor(), edgecolor='none')
plt.show()

print(f"✨ 美化的雷达图已保存为: {output_png}（当前工作目录: {Path.cwd()}）")
print("📊 优化功能包括：")
print("   • 专业的配色方案和渐变效果")
print("   • 成绩等级区域显示（优秀/良好/及格）")
print("   • 平均分环形线和统计信息")
print("   • 高亮最高分和最低分科目")
print("   • 更美观的数据标签和布局")
print("\n💡 如果你要使用自己的文件，请将 filepath 变量改为你的 xlsx 路径，或上传文件并改名为 'scores.xlsx'。")

