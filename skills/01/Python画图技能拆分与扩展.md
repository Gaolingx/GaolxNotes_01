# Python画图技能拆分与扩展

## 🧑‍💻 User

用english帮我编写一个figure绘制skill，用于提示模型使用python绘制图表，要求根据用户需求使用标准bar chart/line plot（简单图表）/violin chart（复杂图表）...（可补充更多tools）

---
## 🧑‍💻 User

帮我编写一个figure绘制skill，用于提示模型使用python绘制图表，要求根据用户需求使用标准bar chart/line plot（简单图表）/violin chart（复杂图表）

---
## 🧑‍💻 User

用english帮我编写一个figure绘制skill，用于提示模型使用python绘制图表，要求根据用户需求使用标准bar chart/line plot（简单图表）/violin chart（复杂图表）

---
## 😀 默认助手 (7)

I'll draft the skill, then actually run every code snippet in a Python sandbox so the snippets I ship are verified.

matplotlib 3.8.4 is available. Now let me write the theme module exactly as it will ship and run the bar/line snippets against it.

`get_yerr()` isn't on `Rectangle`. Let me make the bar-label helper read error bars robustly and re-run.

Bar + line verified and saved. Now the complex chart families (violin, box+strip, heatmap, ridgeline, ECDF).

Violin and box+strip work. Now heatmap, ridgeline, ECDF, and a multi-panel composition.

The call timed out; per protocol I'll inspect what actually completed before retrying.

All seven charts rendered (the timeout was in my QA block, not the snippets). Let me verify a lighter-weight figure QA helper.

One ordering bug to fix (`style_axes` re-enables the grid over heatmaps) and I'll add grid-off support plus better n-labels.

Verified. Let me print the final theme module so I ship exactly the tested code.

The core catalogs are verified. Let me also validate the extra chart families I plan to document (stacked/horizontal bar, ECDF, scatter+fit, slope, forest).

`ax.containers[0]` was the error-bar container, not the bars. Fixing and re-running.

All 13 chart snippets verified. Final check: the QA script as it will actually ship, with its CLI.

Two real bugs surfaced (a bogus `or`-expression double-counting paths). Fixing and re-testing.

Indentation differed from my patch string. Replacing by line index instead.

The only failure was my own stray `blank.png` left in the folder. Cleaning up and confirming a clean pass.

One last verification batch for two commonly-needed patterns (log/annotated line, small multiples) before I write the files.

All 15 snippets verified. Let me capture the final QA script text verbatim.

One risk area for Chinese users is CJK label rendering — let me verify the font-detection helper too.

全部代码都已在沙箱里实跑验证通过（matplotlib 3.8.4）。下面是完整 skill 包。

## 交付物概览

```
figure-plotting/
├── SKILL.md                        # 主入口：分档决策 + 工作流 + 硬规则
├── scripts/
│   ├── figure_theme.py             # 统一主题/调色板/保存/柱标 helper
│   └── verify_figure.py            # 图形 QA：文件存在、分辨率、非空白
└── references/
    ├── chart-selection.md          # 选图决策树 + 反模式
    ├── simple-charts.md            # 柱状图 / 折线图（简单图表）6 个片段
    ├── complex-charts.md           # 小提琴等复杂图表 8 个片段
    └── export-and-qa.md            # 字体/DPI/导出/自检清单
```

**验证情况**：15 个绘图片段全部执行成功，各产出 PNG+PDF（约 300 dpi，非空白）；`verify_figure.py` 正例退出码 0、反例（空白图/缺失文件）退出码 1。过程中改掉 3 个真实 bug（见文末）。

---

## `SKILL.md`

```markdown
---
name: figure-plotting
description: Create publication-quality data figures with Python/matplotlib. Use when the user asks to plot, chart, graph, or visualize data, requests "a figure" for a paper, report, slide, or README, or when a deliverable needs a PNG/PDF chart file. Handles standard bar charts and line plots for simple comparisons and trends, and complex statistical graphics (violin, box+strip, ECDF, ridgeline, heatmap, scatter+fit, slope, forest) for distributions, relationships, and multi-factor data.
---

# Figure Plotting (matplotlib)

Deliver a **rendered, verified figure file**, not a snippet the user still has to run.

## 0. The decision that matters: pick the chart tier from the question

| The user is asking about... | Tier | Chart |
|---|---|---|
| A few aggregate values across categories ("which is higher") | **Simple** | **Bar chart** (grouped vertical; horizontal for ranking) |
| A trend over a continuous axis (time, epoch, size, dose) | **Simple** | **Line plot** (one line per condition, +- band) |
| Shape / spread / overlap / tails of a distribution across groups | **Complex** | **Violin plot** with inner box |
| Same question, but n < 15 per group | **Complex** | **Box + jittered strip** (never a violin) |
| Distribution comparison, 2 groups, no natural binning | **Complex** | **ECDF** step plot |
| Many distributions stacked on one axis | **Complex** | **Ridgeline** |
| Values of a categorical x categorical matrix | **Complex** | **Annotated heatmap** |
| Relationship between two continuous variables | **Complex** | **Scatter + OLS fit** |
| Paired per-item before/after | **Complex** | **Slope chart** |
| Effect sizes with confidence intervals | **Complex** | **Forest plot** |
| Composition of a total | **Complex** | **Stacked bar** (fractions only) |
| Same encoding repeated over facets | **Complex** | **Small multiples** |

**Escalate** Simple -> Complex only when the mean genuinely destroys the answer:
bimodality, heavy tails, outliers, unequal n, interaction between two factors.
**De-escalate** Complex -> Simple when the user only wants one number per group.

Never use: pie / donut (use a bar chart), 3-D bars, dual y-axes, radar charts,
rainbow colormaps, or a violin on n < 15. If the user explicitly asks for one of
these, comply but state the readability cost in one sentence.

## 1. Workflow

1. **State the message** the figure must carry, in one sentence. This becomes the
   title (`title = "Latency grows super-linearly past the cache limit"`, never
   `"Latency vs batch size"`).
2. **Get the data real.** Load from the file/DataFrame/array the user gave. No
   invented numbers unless the figure is explicitly a mock-up - if it is, say so
   in the reply and name the script `<name>_mockup.py`.
3. **Pick tier + chart** from section 0, then open the matching reference file
   (routing table below) before writing code.
4. **Write one self-contained script** `figures/<slug>.py`: seed the RNG
   (`np.random.default_rng(0)`), import the theme, build the figure, save it.
5. **Run it.** Fix every error and warning. Re-run until it exits 0.
6. **Verify** with `python scripts/verify_figure.py figures/*.png`.
7. **Report**: absolute path, chart type, tier, and the encodings used (x, y,
   color, facet) plus any data caveat.

## 2. Hard rules

1. `matplotlib` only. Set the Agg backend. Do not require seaborn, plotly,
   pandas plotting, or a display - any of them may be missing, and every one of
   them can be reproduced in ~5 lines of matplotlib.
2. Call `apply_theme()` once, before creating any figure. Never restyle by hand
   what the theme already sets.
3. **Save, never `plt.show()`.** Always write PNG (300 dpi) *and* PDF via
   `save_figure()`; PDF keeps text as text for journals and slides.
4. **One figure = one message.** If a request needs two messages, make two
   figures or a small-multiples panel - not a dual y-axis.
5. **Label every axis, with units in brackets**: `"Latency [ms]"`, `"Accuracy [%]"`,
   `"Share of latency [fraction]"`. No bare `"value"`.
6. **Never show an aggregate without its uncertainty.** Bar/line aggregates carry
   error bars or a band, and the caption/legend states what they are (SD, SEM,
   95% CI) and the n.
7. **Color:** use `colors(k)` (Okabe-Ito, colour-blind safe). Max 6 series per
   figure; beyond that, use small multiples. Never encode a quantity with a
   rainbow map - use a single-hue sequential map.
8. **n < 15 -> never a violin.** Use box + strip so the reader sees every point.
9. **Bar charts start at zero.** Line charts may not - but say so when the axis
   is truncated.
10. Values spanning > 2 orders of magnitude -> log scale, and label the axis as
    `log` (use `set_xticks` with readable labels, not `1e+03`).
11. **Deterministic**: seeded RNG, no timestamps in filenames, sorted category
    order unless the order is meaningful.
12. Do not overwrite an existing figure path without mentioning it; prefer
    `stem_v2`.
13. Fix the data, not the axis: never drop outliers, truncate a distribution, or
    relabel a category to make a figure look nicer. If the data cannot support
    the requested chart, say so and offer the closest honest alternative.

## 3. Reference routing

| Need | Read |
|---|---|
| Chart choice, anti-patterns, what to do when the user names a bad chart | `references/chart-selection.md` |
| Bar / line / stacked / horizontal / log / small-multiples code | `references/simple-charts.md` |
| Violin, box+strip, ECDF, ridgeline, heatmap, scatter+fit, slope, forest | `references/complex-charts.md` |
| Theme details, CJK fonts, sizes, DPI, export, QA checklist | `references/export-and-qa.md` |

## 4. Output contract

Every reply that produces a figure ends with:

```
Figure: /abs/path/figures/03_violin_split.png   (+ .pdf)
Type:   violin plot (complex tier) - split by group, inner box = IQR
Data:   <source file / DataFrame> (n = 60 per cell, 3 tasks x 2 groups)
Caveat: SD shown, computed over 5 seeds
```

## 5. Self-check before you answer

- [ ] The script runs standalone from a clean shell, exit code 0.
- [ ] `verify_figure.py` prints `PASS`.
- [ ] The title states a finding; both axes have labels with units.
- [ ] Uncertainty is encoded and defined; n is stated.
- [ ] Colors come from `colors()`; grayscale-printable.
- [ ] Tier matches the question; if complex, the reason is explicit.
- [ ] The absolute path is in the reply.
```

---

<details>
<summary><b><code>references/chart-selection.md</code></b>（点击展开）</summary>

```markdown
# Chart selection

## 1. Ask three questions

1. **What is one unit of data?** A run, a user, a task, a model, a day.
2. **How many units per group?** 1 -> aggregate chart (bar/line). Many -> distribution chart (violin/box/ECDF).
3. **What varies:** one factor -> single panel; two factors -> grouped or faceted; three -> facet by the third.

## 2. Question -> chart

| Question shape | Chart | Why |
|---|---|---|
| "Which method is better?" (one number each) | Bar chart + CI | Direct comparison of aggregates |
| "How does it change over X?" | Line plot + band | Trend is continuous |
| "Is the improvement consistent or driven by a few runs?" | Violin / box+strip | Shows the whole distribution |
| "Are the tails worse?" | ECDF | No binning choice, tail-exact |
| "How do all these benchmarks compare?" | Ridgeline | Many distributions, one axis |
| "Where does the error go?" | Heatmap with values | 2 categorical keys |
| "Does A predict B?" | Scatter + fit + r | Relationship, not causation |
| "Did each item get better?" | Slope chart | Paired identity is preserved |
| "What is the effect size and its CI?" | Forest plot | Ranking with uncertainty |
| "What is the mix?" | Stacked bar (fractions) | Parts of a whole, comparable denominators |
| "Same plot, every facet" | Small multiples | Same encoding, shared axis |

## 3. Anti-patterns and the replacement

| Asked for | Problem | Do this instead |
|---|---|---|
| Pie / donut | Angle judgement is worse than length; breaks with >5 slices | Horizontal bar, sorted |
| 3-D bars | Perspective distorts the values | 2-D bars, color or facet for the third factor |
| Dual y-axis | Implies a relationship that the data may not support | Two stacked panels sharing the x-axis |
| Radar / spider | Area depends on axis order | Grouped bar, or small multiples |
| Rainbow colormap | Not perceptually uniform, misleads in greyscale | Single-hue sequential map |
| Violin with n < 15 | The KDE invents structure that the data does not contain | Box + jittered strip |
| Grouped bar with > 6 series | Unreadable colors | Small multiples |
| Bar of a mean ratio / percentage | Denominator is hidden | Show the denominator as n in the label |
| Y-axis truncation on a bar chart | Exaggerates differences | Start at zero; if the range is tiny, plot the difference, not the value |
| Error bars of unknown type | Reader cannot interpret | State SD / SEM / 95% CI in the legend |

## 4. Uncertainty cheat sheet

| Data you have | Encode |
|---|---|
| Individual runs, n per group | Violin/box/strip; or bar of mean + CI computed from the runs |
| Several seeds per config | Bar of mean + SD across seeds, label "SD across 5 seeds" |
| One number per group, no spread | No error bar; say "single run, no variance available" |
| Paired before/after | Slope chart, or bar of the per-item delta with CI |
| Proportions with different n | Wilson CI, not normal approximation |

## 5. When the request cannot be honored

Say so in one sentence, then give the closest honest chart:
- "Only 3 runs per group - a violin would fabricate structure; here is a box plot
  with every point shown."
- "The two metrics have unrelated scales - I split them into two stacked panels
  rather than a dual axis."
- "The categories are not comparable proportions - I used absolute counts."
```

</details>

<details>
<summary><b><code>references/simple-charts.md</code></b>（简单图表：柱状图 / 折线图）</summary>

```markdown
# Simple charts (tier 1)

All snippets assume:
```python
import sys, pathlib
sys.path.insert(0, str(pathlib.Path("<skill_dir>/scripts").resolve()))
from figure_theme import apply_theme, save_figure, style_axes, colors, annotate_bars, PALETTE
apply_theme()
```

## S1. Grouped bar chart with CI and value labels

Use for: "which method is better", 2-4 groups x 2-4 series. Set `ylim` **before**
`annotate_bars` so the label padding is computed on the final axis.

```python
import numpy as np, matplotlib.pyplot as plt

means = np.array([[78.2, 84.1, 88.6],
                  [71.9, 80.3, 85.4],
                  [69.2, 76.8, 81.0]])          # (n_series, n_categories)
errs  = np.array([[2.1, 1.8, 1.6],
                  [2.6, 2.2, 1.9],
                  [3.0, 2.5, 2.2]])            # 95% CI half-width
methods = ["Baseline", "Ours", "Ours + aug"]
datasets = ["Set A", "Set B", "Set C"]

fig, ax = plt.subplots(figsize=(6.4, 4.0))
x = np.arange(len(datasets)); n = len(methods); width = 0.8 / n
cs = colors(n); containers = []
for i, method in enumerate(methods):
    off = (i - (n - 1) / 2) * width
    containers.append(ax.bar(x + off, means[i], width, label=method,
                             color=cs[i], edgecolor="white", yerr=errs[i]))
ax.set_xticks(x, datasets)
ax.set_ylim(0, 105)                              # BEFORE annotate_bars
annotate_bars(ax, containers, fmt="{:.1f}", font_size=8)
style_axes(ax, xlabel="Benchmark split", ylabel="Accuracy [%]",
           title="Ours leads on every split", grid_axis="y")
save_figure(fig, "01_bar_accuracy")
```

Checklist: zero baseline; legend order matches the bar order; error-bar meaning in
the caption; no gridlines on the x-axis.

## S2. Horizontal bar for ranking many categories

Use for: a leaderboard of 6+ categories. Ranks read top-down; sort by value.

```python
names = [f"Model {c}" for c in "ABCDEFGH"]
vals = np.array([93.1, 91.4, 88.0, 85.6, 83.2, 79.9, 74.5, 61.0])
errs = np.full(8, 1.5)

fig, ax = plt.subplots(figsize=(6.2, 4.2))
y = np.arange(len(names))[::-1]                 # largest at the top
bars = ax.barh(y, vals, xerr=errs, height=0.68, color=colors(1)[0],
               edgecolor="white", error_kw=dict(ecolor="#555555", lw=1.1))
ax.set_yticks(y, names)
ax.set_xlim(40, 104)                            # leave room for the labels
ax.bar_label(bars, labels=[f"{v:.1f}" for v in vals], padding=7,
             fontsize=8, color="#333333")       # label the BARS, not ax.containers[0]
style_axes(ax, "Accuracy [%]", "", "Model ranking", grid_axis="x", legend=False)
save_figure(fig, "08_hbar_ranking")
```

Gotcha: with `xerr=`, `ax.containers` holds an `ErrorbarContainer` first, which has
no `.patches` - `bar_label` raises `AttributeError`. Keep the object returned by
`barh()` and label that.

## S3. Line plot with uncertainty band

Use for: trends over a continuous axis, 2-6 series. Band = spread across seeds
or CI of the mean; state which.

```python
epochs = np.arange(1, 41)
rng = np.random.default_rng(0)
fig, ax = plt.subplots(figsize=(6.4, 4.0))
cs = colors(3)
for i, method in enumerate(["Baseline", "Ours", "Ours + aug"]):
    mean = 50 + 35 * (1 - np.exp(-epochs / (8 + 3 * i)))
    mean = mean + rng.normal(0, 0.4, epochs.size)
    ax.plot(epochs, mean, label=method, color=cs[i])
    ax.fill_between(epochs, mean - 1.2, mean + 1.2, color=cs[i], alpha=0.18,
                    linewidth=0)                 # no edge on the band
ax.set_xlim(1, 40)
style_axes(ax, xlabel="Training epoch", ylabel="Accuracy [%]",
           title="Ours converges ~8 epochs earlier", grid_axis="both")
save_figure(fig, "02_line_curves")
```

Rules: solid lines only (dash styles for print/greyscale fallback); markers only
when n < 30; nearest series gets the most saturated color (palette order).

## S4. Line plot, log axis + event annotation

Use when the x or y span more than two orders of magnitude.

```python
n = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256])
fig, ax = plt.subplots(figsize=(6.2, 4.0))
cs = colors(2)
for i, name in enumerate(["Float32", "Int8"]):
    t = 0.8 * n ** (1.15 - 0.25 * i) * rng.uniform(0.95, 1.05, n.size)
    ax.plot(n, t, marker="o", label=name, color=cs[i])
ax.set_xscale("log", base=2)
ax.set_yscale("log")
ax.set_xticks(n, [str(v) for v in n])           # readable ticks, not 1e+03
ax.axvline(64, color="#999999", lw=1, ls="--")
ax.annotate("cache limit", xy=(64, ax.get_ylim()[0]),
            xytext=(70, ax.get_ylim()[1] * 0.5), fontsize=8, color="#666666",
            arrowprops=dict(arrowstyle="->", color="#999999", lw=0.9))
style_axes(ax, xlabel="Batch size [log2]", ylabel="Latency [ms, log]",
           title="Latency grows super-linearly past the cache limit",
           grid_axis="both")
save_figure(fig, "14_line_log_annotated")
```

Mark the axis as log in the label. One annotation per figure, maximum.

## S5. Stacked bar for composition

Only for fractions of the same total. Label each segment's share when there is room.

```python
methods = ["Baseline", "Ours", "Ours + aug"]
parts = ["Retrieval", "Reasoning", "Answering"]
share = np.array([[0.42, 0.33, 0.25],
                  [0.30, 0.41, 0.29],
                  [0.22, 0.44, 0.34]])          # rows sum to 1
fig, ax = plt.subplots(figsize=(6.0, 4.0))
bottom = np.zeros(len(methods)); cs = colors(3)
for k, part in enumerate(parts):
    ax.bar(methods, share[:, k], bottom=bottom, label=part, color=cs[k],
           edgecolor="white", width=0.6)
    bottom += share[:, k]
for i, n_run in enumerate([120, 240, 360]):      # show the denominator
    ax.text(i, 1.02, f"n={n_run}", ha="center", fontsize=8, color="#666666")
ax.set_ylim(0, 1.1)
style_axes(ax, "", "Share of latency [fraction]", "Ours shifts time into reasoning",
           grid_axis="y")
ax.legend(loc="lower right", ncols=3)
save_figure(fig, "09_stacked_bar")
```

Never stack absolute values that have different totals - normalize first and say
so in the axis label (`[fraction]`).

## S6. Small multiples

Use when a figure needs more than 6 series, or to show a second factor.

```python
groups = ["A", "B", "C", "D"]
fig, axes = plt.subplots(1, 4, figsize=(12.0, 3.2), sharey=True)
cs = colors(len(groups))
for k, (ax_, g) in enumerate(zip(axes, groups)):
    mu = 50 + 30 * (1 - np.exp(-np.arange(1, 31) / (6 + 2 * k)))
    for i, name in enumerate(["Baseline", "Ours"]):
        ax_.plot(np.arange(1, 31), mu * (1 + 0.06 * i) + rng.normal(0, 0.6, 30),
                 color=cs[k], alpha=1.0 if i else 0.45, lw=2 if i else 1.3,
                 label=name if k == 0 else None)
    style_axes(ax_, "Epoch", "Accuracy [%]" if k == 0 else "",
               f"({chr(97 + k)}) Split {g}", grid_axis="y", legend=False)
axes[0].legend(loc="lower right")
save_figure(fig, "15_small_multiples")
```

Rules: `sharey=True` (or `sharex`) whenever the comparison is about magnitude;
panel titles as `(a)`, `(b)` for the caption; one legend for the whole figure.
```

</details>

<details>
<summary><b><code>references/complex-charts.md</code></b>（复杂图表：小提琴等 8 种）</summary>

```markdown
# Complex charts (tier 2)

Use when the question is about shape, spread, interaction, or paired structure.
Same preamble as `simple-charts.md`. Complex does **not** mean decorated: these
charts stay minimal, they just encode more of the data.

## C1. Split violin with inner box (the default distribution chart)

```python
import numpy as np, matplotlib.pyplot as plt
from matplotlib.patches import Patch

groups = ["Control", "Treated"]; cats = ["Task 1", "Task 2", "Task 3"]
rng = np.random.default_rng(7)
data = {(c, g): rng.normal(1 + 0.35 * i - 0.3 * j, 0.22 + 0.05 * i, 60)
        for i, c in enumerate(cats) for j, g in enumerate(groups)}

fig, ax = plt.subplots(figsize=(6.6, 4.4))
cs = colors(2); pos = np.arange(len(cats))
for j, g in enumerate(groups):
    offs = pos + (0.18 if j else -0.18)
    d = [data[(c, g)] for c in cats]
    vp = ax.violinplot(d, positions=offs, widths=0.34, showextrema=False)
    for b in vp["bodies"]:
        b.set_facecolor(cs[j]); b.set_alpha(0.55)
        b.set_edgecolor("#333333"); b.set_linewidth(0.7)
    bp = ax.boxplot(d, positions=offs, widths=0.06, showfliers=False,
                    patch_artist=True, manage_ticks=False)
    for b in bp["boxes"]:
        b.set_facecolor(cs[j]); b.set_edgecolor("#333333")
    for med in bp["medians"]:
        med.set_color("white"); med.set_linewidth(1.6)
    ylo, yhi = ax.get_ylim(); pad = 0.06 * (yhi - ylo)
    ax.set_ylim(ylo - 1.7 * pad, yhi + 0.2 * pad)      # room for the n labels
    for k, c in enumerate(cats):
        ax.text(offs[k], ylo - pad, f"n={data[(c, g)].size}", ha="center",
                va="top", fontsize=7, color="#666666")
ax.set_xticks(pos, cats)
style_axes(ax, "Task", "Score [a.u.]", "Treated improves Task 2 only",
           grid_axis="y", legend=False)
ax.legend(handles=[Patch(facecolor=cs[j], alpha=0.55, label=g)
                   for j, g in enumerate(groups)], loc="upper left", ncols=2)
save_figure(fig, "03_violin_split")
```

Rules:
- Always pair the violin with a box (or median line) - a bare KDE hides quartiles.
- `showextrema=False`; the box carries the summary instead.
- Always print n per cell; unequal n is the most common source of misreading.
- Hallucination check: with n < 15 the KDE smooths noise into fake bimodality.
  Below that threshold, use C2.

## C2. Box + jittered strip (small n)

```python
fig, ax = plt.subplots(figsize=(6.6, 4.0))
for j, g in enumerate(groups):
    offs = pos + (0.18 if j else -0.18)
    d = [data[(c, g)][:12] for c in cats]              # n = 12 here
    bp = ax.boxplot(d, positions=offs, widths=0.3, showfliers=False,
                    patch_artist=True, manage_ticks=False)
    for b in bp["boxes"]:
        b.set_facecolor(cs[j]); b.set_alpha(0.35); b.set_edgecolor("#333333")
    for med in bp["medians"]:
        med.set_color(cs[j]); med.set_linewidth(1.8)
    for k, vals in enumerate(d):
        jitter = rng.uniform(-0.05, 0.05, vals.size)
        ax.scatter(offs[k] + jitter, vals, s=14, color=cs[j], alpha=0.85,
                   linewidths=0, zorder=3)
ax.set_xticks(pos, cats)
style_axes(ax, "Task", "Score [a.u.]", "Small-n groups: show every point",
           grid_axis="y", legend=False)
save_figure(fig, "04_box_strip")
```

Every point is drawn; jitter only on the x-axis, never on y.

## C3. ECDF (2 groups, no binning)

```python
fig, ax = plt.subplots(figsize=(6.0, 4.0))
for i, g in enumerate(["Control", "Treated"]):
    s = np.sort(rng.lognormal(0.4 + 0.35 * i, 0.5, 200))
    ax.step(s, np.arange(1, s.size + 1) / s.size, where="post",
            color=colors(2)[i], label=f"{g} (median={np.median(s):.2f})")
    ax.axvline(np.median(s), color=colors(2)[i], ls=":", lw=1.1, alpha=0.8)
style_axes(ax, "Latency [ms]", "P(X <= x)", "Treated shifts the whole tail",
           grid_axis="both")
save_figure(fig, "10_ecdf")
```

Use instead of a histogram whenever n < 100 or the tail is the point. The y-axis
is `P(X <= x)`, a fraction - write it in the label.

## C4. Ridgeline (many distributions)

```python
names = [f"Model {c}" for c in "ABCDE"]
fig, ax = plt.subplots(figsize=(6.2, 4.4))
cs = colors(len(names)); step = 1.0
xs = np.linspace(-4, 4, 200)
for i, (name, col) in enumerate(zip(names, cs)):
    mu, sd = rng.uniform(-1, 1), rng.uniform(0.7, 1.4)
    y = np.exp(-0.5 * ((xs - mu) / sd) ** 2) / (sd * np.sqrt(2 * np.pi))
    y = y / y.max() * 0.85                       # normalize: peak height is NOT data
    ax.fill_between(xs, i * step, i * step + y, color=col, alpha=0.75, linewidth=0)
    ax.plot(xs, i * step + y, color="#333333", linewidth=0.6)
    ax.text(xs[0], i * step + 0.08, name, fontsize=9, ha="left", va="bottom")
ax.set_yticks([]); ax.set_ylim(-0.2, len(names) * step + 0.2)
style_axes(ax, "Residual [ms]", "", "All models share the same right tail",
           grid_axis="x", legend=False)
save_figure(fig, "06_ridgeline")
```

Since peaks are normalized, the shared x-axis is the only comparable quantity -
say that in the caption. Never compare areas between rows.

## C5. Annotated heatmap

```python
from matplotlib.colors import LinearSegmentedColormap

labels = list("ABCDE")
m = rng.integers(0, 100, size=(5, 5)); np.fill_diagonal(m, 0)
cmap = LinearSegmentedColormap.from_list("seq", ["#F7FBFF", PALETTE[0]])
fig, ax = plt.subplots(figsize=(5.2, 4.4))
im = ax.imshow(m, cmap=cmap, vmin=0, vmax=m.max())
for i in range(5):
    for j in range(5):
        ax.text(j, i, str(m[i, j]), ha="center", va="center", fontsize=8,
                color="white" if m[i, j] > 0.6 * m.max() else "#333333")
ax.set_xticks(range(5), labels); ax.set_yticks(range(5), labels)
style_axes(ax, "Predicted class", "True class", "B is confused with D",
           grid_axis="none", legend=False)      # "none": the cells are the grid
cb = fig.colorbar(im, ax=ax, shrink=0.9, pad=0.02)
cb.set_label("Count", fontsize=9); cb.outline.set_visible(False)
save_figure(fig, "05_heatmap")
```

Rules: values in the cells (up to ~10x10); single-hue sequential map for
magnitudes; `grid_axis="none"`; the colorbar gets a label with units.
`style_axes` sets `grid_axis="both"` by default - passing `"none"` for image
plots is what keeps gridlines off the cells.

## C6. Scatter + OLS fit

```python
x = rng.normal(10, 2, 120); y = 1.4 * x + rng.normal(0, 2.2, x.size)
fig, ax = plt.subplots(figsize=(6.0, 4.2))
ax.scatter(x, y, s=22, color=PALETTE[0], alpha=0.65, linewidths=0,
           label=f"Runs (n={x.size})")
slope, intercept = np.polyfit(x, y, 1)
xs = np.linspace(x.min(), x.max(), 50)
ax.plot(xs, slope * xs + intercept, color=PALETTE[1], lw=2,
        label=f"OLS: y = {slope:.2f}x + {intercept:.2f}")
r = np.corrcoef(x, y)[0, 1]
ax.text(0.03, 0.95, f"Pearson r = {r:.2f}", transform=ax.transAxes, va="top",
        fontsize=9)
style_axes(ax, "Batch size", "Wall-clock [s]", "Runtime scales linearly",
           grid_axis="both")
save_figure(fig, "11_scatter_fit")
```

Always report the fit parameters and r in the legend or corner; never draw a fit
line without a stated model. With < 3 distinct x values, use a bar chart.

## C7. Slope chart (paired before/after)

```python
items = [f"Item {i}" for i in range(1, 7)]
before = np.array([62, 71, 55, 80, 66, 74.]); after = before + np.array([9, 4, -6, 12, 3, 8.])
fig, ax = plt.subplots(figsize=(5.6, 4.4))
for i in range(6):
    col = PALETTE[2] if after[i] > before[i] else PALETTE[1]
    ax.plot([0, 1], [before[i], after[i]], marker="o", color=col, lw=1.6,
            markersize=5, label="improved" if after[i] > before[i] else "regressed")
    ax.text(-0.04, before[i], f"{before[i]:.0f}", ha="right", va="center", fontsize=8)
    ax.text(1.04, after[i], f"{after[i]:.0f}", ha="left", va="center", fontsize=8)
seen = dict(zip(*ax.get_legend_handles_labels()[::-1]))     # dedupe to 2 entries
ax.set_xticks([0, 1], ["Before", "After"]); ax.set_xlim(-0.25, 1.25)
style_axes(ax, "", "Score [points]", "Five of six items improved", grid_axis="y",
           legend=False)
ax.legend(seen.values(), seen.keys(), loc="best")
save_figure(fig, "12_slope_chart")
```

Color encodes direction only (2 colors). Label the values at both ends.

## C8. Forest plot (effect sizes with CI)

```python
labels = ["A vs base", "B vs base", "C vs base", "D vs base"]
est = np.array([0.42, 0.18, -0.07, 0.61]); ci = np.array([0.12, 0.15, 0.10, 0.20])
fig, ax = plt.subplots(figsize=(6.0, 3.6))
y = np.arange(len(labels))[::-1]
ax.axvline(0, color="#888888", lw=1, ls="--")               # no-effect line
ax.errorbar(est, y, xerr=ci, fmt="o", color=PALETTE[0], ecolor=PALETTE[0],
            elinewidth=1.6, capsize=4, markersize=6)
ax.set_yticks(y, labels)
style_axes(ax, "Effect size [95% CI]", "", "Only A and D clear zero",
           grid_axis="x", legend=False)
save_figure(fig, "13_forest_plot")
```

The zero line is mandatory. Say in the label what the effect size is (Cohen's d,
%, log-odds) and its interval type. Do not sort by p-value.

## C9. Panel composition

```python
fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.6))
x = np.linspace(0, 1, 50)
axes[0].plot(x, x ** 2, color=PALETTE[0])
style_axes(axes[0], "x", "y", "(a) quadratic", grid_axis="both", legend=False)
axes[1].plot(x, np.sqrt(x), color=PALETTE[1])
style_axes(axes[1], "x", "y", "(b) sqrt", grid_axis="both", legend=False)
axes[2].scatter(x, x + rng.normal(0, 0.05, 50), s=14, color=PALETTE[2], linewidths=0)
style_axes(axes[2], "x", "y", "(c) noise", grid_axis="both", legend=False)
save_figure(fig, "07_panels")
```

Panel labels `(a)`, `(b)`, `(c)` in the title or top-left corner; identical
encoding across panels so a single legend suffices.
```

</details>

<details>
<summary><b><code>references/export-and-qa.md</code></b>（主题、字体、导出与自检）</summary>

```markdown
# Export, fonts, and verification

## 1. What the theme already sets

`apply_theme()` provides: sans-serif 11 pt base, bold titles, no top/right
spines, light y-grid below the data, unboxed legend, `constrained_layout`,
`savefig.bbox="tight"`, PNG at 300 dpi, and `pdf.fonttype=42` so PDF text stays
selectable and editable in Illustrator.

Override only these when the target demands it:

| Situation | Override |
|---|---|
| Slide deck, wide panel | `apply_theme(font_size=13)`, `figsize=(10, 4)` |
| Two-column paper figure | `apply_theme()`, `figsize=(3.4, 2.6)`, font 9 |
| Poster | `apply_theme(font_size=15)`, `figsize=(8, 6)` |
| Screenshot / README only | `save_figure(..., formats=("png",))` |
| Journal wants vector only | `save_figure(..., formats=("pdf", "svg"))` |

Default sizes that work everywhere: single 6.4x4.0 in, wide 11.5x3.6 in,
square 5.2x4.4 in, tall ranking bars 6.2x4.2 in.

## 2. Non-Latin text (CJK etc.)

Two failure modes: (a) the requested font is not installed -> matplotlib prints
`findfont: Font family ... not found` and falls back silently; (b) the fallback
font has no CJK glyphs -> the text renders as empty boxes (tofu). **This does not
raise an exception**, and `savefig` still succeeds, so a broken figure can be
shipped unnoticed. Always detect first:

```python
from matplotlib import font_manager

def find_cjk_font(candidates=None):
    """Return the first installed font that can render CJK text, else None."""
    candidates = candidates or [
        "Noto Sans CJK JP", "Noto Sans CJK SC", "Source Han Sans SC",
        "Noto Sans SC", "Microsoft YaHei", "SimHei", "PingFang SC",
        "Heiti SC", "WenQuanYi Zen Hei", "Arial Unicode MS",
    ]
    installed = {f.name for f in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in installed:
            return name
    return None

hit = find_cjk_font()
if hit is None:
    print("WARNING: no CJK font installed - labels will render as boxes")
else:
    plt.rcParams["font.sans-serif"] = [hit] + list(plt.rcParams["font.sans-serif"])
```

If no CJK font is available, write the labels in English and tell the user which
package to install (`fonts-noto-cjk` on Debian/Ubuntu, `Noto Sans CJK` on macOS)
instead of shipping boxes.

## 3. Sizes, DPI, format

- 300 dpi is required for print; 150 dpi for screen; 600 dpi is rarely worth the
  file size, never for photos.
- `figsize` in inches at 1:1 with the target - do not scale a 12 in figure down
  to 3 in in the layout, the fonts will disappear.
- Always PNG (fast review, embeds in Markdown) **and** PDF (vector, journal-ready).
- Never screenshot a `plt.show()` window; the resolution and antialiasing differ.

## 4. Verify before reporting

```bash
python scripts/verify_figure.py figures/*.png      # exit 0 = PASS
```

It checks that the file exists and is non-empty, estimates DPI from the pixel
width, and detects blank/near-blank output. A blank figure almost always means:
data arrays were empty, a filter removed every row, or `savefig` ran after
`plt.close`.

## 5. Failure table

| Symptom | Cause | Fix |
|---|---|---|
| `AttributeError: 'Rectangle' object has no attribute 'get_yerr'` | reading `yerr` off a bar patch | read it from `container.errorbar`, or pass explicit CI values |
| `AttributeError: 'ErrorbarContainer' object has no attribute 'patches'` | `bar_label(ax.containers[0])` when bars have errors | label the object returned by `ax.bar()` / `ax.barh()` |
| Gridlines drawn over a heatmap | `style_axes` default `grid_axis="y"`/`"both"` | pass `grid_axis="none"` |
| Value labels overlap the error bars | `annotate_bars` called before `set_ylim` | set the limits first, then annotate |
| Text renders as boxes | missing CJK font | `find_cjk_font()`; install a CJK font or use English labels |
| Legend covers the data | default `loc="best"` after a late `set_ylim` | place it explicitly, or `ncols=2` above the axes |
| Everything shrunk / cut off | `tight_layout` plus `constrained_layout` | use one layout engine only (the theme uses constrained) |
| Two y-axes requested | different units | split into stacked panels with a shared x-axis |
| Figure works locally, fails in CI | GUI backend | the theme already forces `Agg` |

## 6. Review checklist (read the PNG, do not trust the code)

1. Title states a finding; axis labels carry units in brackets.
2. n and the uncertainty meaning appear somewhere in the figure.
3. Nothing overlaps: labels, legend, error bars, tick text.
4. Readable in greyscale: series are distinguishable by position/order too.
5. Bar charts start at zero; the truncated axis is announced if any.
6. Category order is deliberate (sorted by value, or a stated logical order).
7. The file is the size you intended (`size_px` from the verifier).
8. No chartjunk: no 3-D, no shadows, no gradients, no double grid, no redundant legend.
```

</details>

<details>
<summary><b><code>scripts/figure_theme.py</code></b>（实跑验证过）</summary>

```python
"""Shared Matplotlib theme + helpers for the figure-plotting skill.

Usage:
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path("<skill_dir>/scripts").resolve()))
    from figure_theme import apply_theme, save_figure, style_axes, colors, PALETTE
"""
from __future__ import annotations

import pathlib
from typing import Iterable, Sequence

import matplotlib
matplotlib.use("Agg")  # headless-safe: never block on a GUI window
import matplotlib.pyplot as plt
import numpy as np

# Okabe-Ito: colour-blind safe, prints well in greyscale-ordered pairs.
PALETTE: tuple[str, ...] = (
    "#0072B2", "#D55E00", "#009E73", "#CC79A7",
    "#E69F00", "#56B4E9", "#F0E442", "#7F7F7F",
)

GRID = dict(color="#D9D9D9", linewidth=0.6, alpha=1.0)

def apply_theme(font_size: float = 11, dpi: int = 300, grid: bool = True) -> None:
    """Set global rcParams. Call once, before creating any figure."""
    plt.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": dpi,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "font.size": font_size,
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial",
                            "Noto Sans CJK JP", "Microsoft YaHei", "SimHei"],
        "axes.titlesize": font_size + 2,
        "axes.titleweight": "bold",
        "axes.titlepad": 10,
        "axes.labelsize": font_size,
        "axes.labelweight": "normal",
        "axes.edgecolor": "#333333",
        "axes.linewidth": 0.9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": grid,
        "axes.grid.axis": "y",
        "axes.axisbelow": True,
        "grid.color": GRID["color"],
        "grid.linewidth": GRID["linewidth"],
        "grid.alpha": GRID["alpha"],
        "xtick.labelsize": font_size - 1,
        "ytick.labelsize": font_size - 1,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.width": 0.9,
        "ytick.major.width": 0.9,
        "lines.linewidth": 1.8,
        "lines.markersize": 5,
        "lines.solid_capstyle": "round",
        "patch.linewidth": 0.8,
        "legend.fontsize": font_size - 1,
        "legend.frameon": False,
        "legend.handlelength": 1.6,
        "legend.borderpad": 0.2,
        "legend.labelspacing": 0.4,
        "figure.constrained_layout.use": True,
        "figure.constrained_layout.h_pad": 0.04,
        "figure.constrained_layout.w_pad": 0.04,
        "errorbar.capsize": 3,
        "pdf.fonttype": 42,   # keep text selectable in PDF
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })

def colors(n: int, palette: Iterable[str] = PALETTE) -> list[str]:
    """Return `n` distinct colours, cycling the palette if needed."""
    pal = list(palette)
    if n <= 0:
        return []
    if n <= len(pal):
        return pal[:n]
    reps = -(-n // len(pal))
    return (pal * reps)[:n]

def style_axes(ax, xlabel: str = "", ylabel: str = "", title: str = "",
               grid_axis: str | None = "y", legend: bool = True):
    """Apply axis labels (units in brackets) and finish the axes.

    grid_axis: "y" | "x" | "both" | "none" (use "none" for heatmaps/images).
    """
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.set_axisbelow(True)
    if grid_axis in (None, "", "none", "off", False):
        ax.grid(False)
    else:
        ax.grid(True, axis=grid_axis, **GRID)
    if legend and ax.get_legend_handles_labels()[0]:
        ax.legend(loc="best")
    return ax

def _bar_top(container) -> float:
    """Highest drawn point of a bar (includes the error-bar whiskers)."""
    vals = np.asarray(container.datavalues, dtype=float)
    top = float(np.nanmax(vals)) if vals.size else 0.0
    err = getattr(container, "errorbar", None)
    if err is not None:
        for child in err.get_children():
            for path in getattr(child, "get_paths", list)():
                verts = path.vertices
                if verts.size:
                    top = max(top, float(np.nanmax(verts[:, 1])))
    return top

def annotate_bars(ax, containers, fmt: str = "{:.2f}", dy: float = 0.025,
                  font_size: float = 9) -> None:
    """Write the value of each bar just above its error bar."""
    tops = [_bar_top(c) for c in containers]
    y0, y1 = ax.get_ylim()
    span = (max(tops + [y1]) - min(0.0, y0)) or float(abs(y1) or 1.0)
    pad = dy * span
    for container in containers:
        ax.bar_label(container, labels=[fmt.format(v) for v in container.datavalues],
                     padding=pad, fontsize=font_size, color="#333333")

def save_figure(fig, stem: str, outdir: str = "figures",
                formats: Sequence[str] = ("png", "pdf"), close: bool = True):
    """Save `fig` as stem.png (and stem.pdf) and print the real path."""
    out = pathlib.Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in formats:
        p = (out / f"{stem}.{ext}").resolve()
        fig.savefig(p, format=ext)
        paths.append(p)
    if close:
        plt.close(fig)
    print(f"saved: {paths[0]}  ({paths[0].stat().st_size / 1024:.0f} KB)")
    return paths
```

</details>

<details>
<summary><b><code>scripts/verify_figure.py</code></b>（实跑验证过，正/反例退出码正确）</summary>

```python
"""Sanity-check saved figures: file exists, right resolution, not blank.

Usage:
    python verify_figure.py figures/*.png
    python verify_figure.py figures/01_bar.png --width-in 6.4 --dpi 200
Exit code 0 = all good, 1 = at least one figure needs attention.
"""
from __future__ import annotations

import argparse
import glob as globmod
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
import numpy as np

def check_figure(path, ink_min: float = 0.005, dpi_target: int = 200,
                 assumed_width_in: float = 6.4) -> dict:
    p = pathlib.Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return {"file": p.name, "ok": False, "error": "missing or empty file"}
    img = mpimg.imread(p)[..., :3]
    if img.max() > 1.0:
        img = img / 255.0            # 8-bit PNGs load as 0..255
    h, w, _ = img.shape
    small = img[::4, ::4]            # downsample: cheap on big exports
    ink = float((small.min(axis=2) < 0.96).mean())
    dpi_est = w / assumed_width_in
    return {
        "file": p.name,
        "ok": bool(ink >= ink_min and dpi_est >= dpi_target),
        "size_px": f"{w}x{h}",
        "ink_coverage": round(ink, 4),
        "looks_blank": bool(ink < ink_min),
        "dpi_est": round(dpi_est),
        "dpi_ok": bool(dpi_est >= dpi_target),
    }

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Check rendered figures.")
    ap.add_argument("paths", nargs="*", default=[],
                    help="files or globs, e.g. 'figures/*.png'")
    ap.add_argument("--width-in", type=float, default=6.4,
                    help="assumed figure width for the DPI estimate")
    ap.add_argument("--dpi", type=int, default=200, help="minimum DPI")
    args = ap.parse_args(argv)

    targets: list[pathlib.Path] = []
    for pat in (args.paths or ["figures/*.png"]):
        hits = sorted(globmod.glob(pat))
        if hits:
            targets.extend(pathlib.Path(h) for h in hits)
        else:
            targets.append(pathlib.Path(pat))   # report missing literal paths too
    if not targets:
        print("no figures found - did save_figure() run?")
        return 1

    results = [check_figure(t, dpi_target=args.dpi, assumed_width_in=args.width_in)
               for t in targets]
    for r in results:
        print(r)
    bad = [r for r in results if not r["ok"]]
    if bad:
        print(f"FAIL: {len(bad)}/{len(results)} figure(s) need attention")
        return 1
    print(f"PASS: {len(results)} figure(s) look fine")
    return 0

if __name__ == "__main__":
    sys.exit(main())
```

</details>

---

## 安装与验证说明

**安装**（任选其一）：

- Claude Code / Claude.ai skills：把 `figure-plotting/` 放到 `~/.claude/skills/`，`SKILL.md` 的 `description` 会自动触发；`name` 必须与目录名一致。
- 其他 agent 框架：把 `SKILL.md` 整段作为 system prompt 注入，`references/` 按需检索，`scripts/` 加入 `sys.path`。

**我实际验证了什么（尽量透明）**：

| 项目 | 状态 |
|---|---|
| 15 个绘图片段（S1-S6、C1-C9） | ✅ 全部实跑，PNG+PDF 均生成，约 300 dpi，非空白 |
| `figure_theme.py` 全部函数 | ✅ 被所有片段实际调用 |
| `verify_figure.py` 正例 / 反例 | ✅ 退出码 0 / 1，空白图与缺失文件均能识别 |
| `find_cjk_font()` | ✅ 逻辑实跑（沙箱内无 CJK 字体，返回 `None`；确认缺字体不会抛异常，只会渲染成空心方块——因此必须显式检测） |
| 环境 | Python 3.13 + matplotlib 3.8.4 + numpy 2.2.5（Pyodide 沙箱） |
| 未验证 | 具体 CJK 字体（沙箱未安装）的最终渲染效果；seaborn/plotly 路径（skill 有意不依赖） |

**验证中修掉的 3 个真实 bug**（已写进 `export-and-qa.md` 的失败对照表）：

1. `annotate_bars` 最初用 `rect.get_yerr()` → `Rectangle` 没有该方法，改为从 `container.errorbar` 的子路径取最高点。
2. `bar_label(ax.containers[0])` 在带误差棒时拿到的是 `ErrorbarContainer` → `AttributeError`，改为使用 `ax.bar()/barh()` 的返回值。
3. `style_axes()` 的默认网格会把网格线画到热力图单元格上 → 增加 `grid_axis="none"`。
