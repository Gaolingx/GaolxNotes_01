# AV1 与 H.264/HEVC QP 近似换算器

## 🧑‍💻 User

```
#### 4.5.1 AV1 的 QP 标度：不可按数字换算

**结论先说**：H.264 与 HEVC 共用 **0–51** 标度（数字可互相参考，HEVC 同数值略更狠）；**AV1 是 0–255 的 qindex，是完全另一把尺子**。"H.264 用 38，AV1 也填 38"是错的，而且是**错向"更严"**的方向 —— 超发更厉害。

**为什么不能按比例换算**：QP 与量化步长是**指数**关系（约 `2^((QP-4)/6)`），而 NVENC 的 AV1 qindex 并非把 0–51 线性拉长：**低段步子细、高段步子粗**。于是"×5"这条常见经验**只在高端近似成立**：

| H.264/HEVC QP | 28 | 34 | 38 | 42 | 47 | 51 |
|---|---|---|---|---|---|---|
| 字节等价的 AV1 qindex | **≈28** | **≈100** | **≈142** | **≈178** | **≈215** | **≈242** |
| 比值（AV1 : H.264） | 1.0 | 2.9 | 3.7 | 4.2 | 4.6 | 4.7 |

（若用 ×5 去换算 H.264 的 `34`，会写成 `170`，而等价值只有 ≈100 → **上限会比你以为的松得多**。）

**实测依据**（同片源、`rc=constqp`、同 preset，平均字节/帧，720p）：

困难内容（逐帧随机噪声，最坏情况）：

| H.264 QP | 19 | 23 | 28 | 34 | 38 | 42 | 47 | 51 |
|---|---|---|---|---|---|---|---|---|
| H.264 B/帧 | 45651 | 37555 | 28049 | 18765 | 16087 | 13320 | 10428 | 8215 |
| HEVC B/帧 | 27312 | 24979 | 21697 | 18135 | 15980 | 13061 | 10152 | 7959 |

| AV1 qindex | 23 | 51 | 76 | 92 | 102 | 128 | 138 | 152 | 176 | 204 | 230 | 255 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| AV1 B/帧 | 29200 | 22784 | 20259 | 19192 | 18644 | 17120 | 16385 | 15356 | 13511 | 11359 | 9258 | 7270 |

简单内容（渐变 + 移动方块）：

| H.264 QP | 19 | 23 | 28 | 34 | 38 | 42 | 47 | 51 |
|---|---|---|---|---|---|---|---|---|
| H.264 B/帧 | 255 | 209 | 160 | 121 | 105 | 112 | 106 | 106 |

| AV1 qindex | 23 | 51 | 76 | 92 | 102 | 128 | 138 | 152 | 176 | 204 | 230 | 255 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| AV1 B/帧 | 890 | 292 | 234 | 181 | 197 | 125 | 119 | 136 | 140 | 96 | 85 | 89 |

两个直接推论：

- **AV1 的可用下限段（qindex ≈23）在简单内容上比 H.264 的 QP 19 还精细**（890 B vs 255 B）—— **AV1 低段基本是留给近无损场景的**，把 H.264 的低数值抄过去没有意义。
- **AV1 的最粗糙端（255）比 H.264 的最粗糙端（51）还要狠**（困难内容 7270 B vs 8215 B；简单内容 89 B vs 106 B）。所以**上限侧的默认值是等价的**（都是 100% 标度），**下限侧不是**。

**实践规则（三条）**：

1. **禁止换算，只认实测。** 改 AV1 的 QP 上下限时，用 [§5](#5-自检怎么确认参数真的生效了) 的 `qpmax=` 后缀确认已下发，再看帧大小/画质收敛。
2. **粗粒度起步值**：想要"H.264 38 的效果"→ AV1 给 **≈140**；想要"H.264 47"→ AV1 给 **≈215**。数字只能当起点。
3. **注意默认值位置**：`min_qp_h264 = 19`（标度 37%）、`min_qp_hevc = 23`（45%）、`min_qp_av1 = 23`（**9%**）。所以"打开 `nvenc_enable_min_qp` 但不动数值"在 **AV1 上几乎等于没设**。
```
html编写一个av1与hevc/h264 qp近似计算器

---
## 😀 默认助手 (7)

这是可直接运行的HTML换算器，基于您提供的实测数据实现了AV1与H.264/HEVC QP的双向近似计算，并附曲线对比与字节验证。
```html
<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>AV1 ⇄ H.264/HEVC QP 近似换算器</title>
<style>
:root{
  --bg:#0e1116; --panel:#151b23; --panel2:#1a222c; --line:#26313d;
  --txt:#e8eef5; --mute:#8b98a8;
  --av1:#4da3ff; --h26:#ffb454; --ok:#3ddc97; --warn:#ff7a59; --hevc:#b98cff;
}
*{box-sizing:border-box}
html,body{margin:0;padding:0}
body{
  background:radial-gradient(1100px 560px at 18% -12%, #17222d 0%, var(--bg) 62%) fixed, var(--bg);
  color:var(--txt);
  font-family:ui-sans-serif,-apple-system,"Segoe UI",Roboto,"Noto Sans SC","PingFang SC","Microsoft YaHei",sans-serif;
  line-height:1.65; padding:26px 16px 70px;
  -webkit-font-smoothing:antialiased;
}
.wrap{max-width:1020px;margin:0 auto}
h1{font-size:24px;line-height:1.35;margin:0 0 8px;letter-spacing:.2px}
h1 em{font-style:normal;color:var(--av1)}
.sub{margin:0 0 6px;color:var(--mute);font-size:14px}
.sub strong{color:var(--warn);font-weight:600}
.card{
  background:linear-gradient(180deg,var(--panel2),var(--panel));
  border:1px solid var(--line); border-radius:14px;
  padding:18px 20px; margin:16px 0;
  box-shadow:0 10px 30px -18px #000;
}
.card h2{font-size:14px;margin:0 0 14px;color:var(--mute);font-weight:600;letter-spacing:.08em;text-transform:uppercase}

/* ---------- 双栏输入 ---------- */
.grid2{display:grid;grid-template-columns:1fr 46px 1fr;align-items:center;gap:10px}
.side label{display:flex;justify-content:space-between;align-items:baseline;font-size:13px;margin-bottom:8px;color:var(--mute)}
.side label b{font-size:14px;letter-spacing:.02em}
.side.h label b{color:var(--h26)}
.side.a label b{color:var(--av1)}
.side label span{font-size:11px;color:#606f80}
.numrow{display:flex;gap:10px;align-items:center}
input[type=number]{
  width:104px;background:#0d1218;border:1px solid var(--line);color:var(--txt);
  border-radius:10px;padding:10px 12px;font-size:22px;font-weight:600;
  font-variant-numeric:tabular-nums;text-align:center;outline:none;
}
input[type=number]:focus{border-color:var(--av1);box-shadow:0 0 0 3px rgba(77,163,255,.14)}
.side.h input[type=number]{color:var(--h26)}
.side.a input[type=number]{color:var(--av1)}
input[type=range]{
  -webkit-appearance:none;appearance:none;width:100%;height:26px;background:transparent;margin:8px 0 2px;cursor:pointer;
}
input[type=range]::-webkit-slider-runnable-track{height:6px;border-radius:99px;background:#0b1016;border:1px solid var(--line)}
input[type=range]::-webkit-slider-thumb{
  -webkit-appearance:none;width:18px;height:18px;border-radius:50%;margin-top:-7px;
  background:var(--av1);border:2px solid #08121d;box-shadow:0 0 0 3px rgba(77,163,255,.18);
}
.side.h input[type=range]::-webkit-slider-thumb{background:var(--h26);box-shadow:0 0 0 3px rgba(255,180,84,.18)}
input[type=range]::-moz-range-track{height:6px;border-radius:99px;background:#0b1016;border:1px solid var(--line)}
input[type=range]::-moz-range-thumb{width:16px;height:16px;border:2px solid #08121d;border-radius:50%;background:var(--av1)}
.side.h input[type=range]::-moz-range-thumb{background:var(--h26)}
.pct{font-size:12px;color:#606f80;font-variant-numeric:tabular-nums}
.swap{font-size:22px;color:#3c4a5a;text-align:center;user-select:none}

/* ---------- 结果条 ---------- */
.result{display:flex;flex-wrap:wrap;gap:18px 26px;align-items:center}
.big{display:flex;align-items:baseline;gap:8px}
.big .k{font-size:12px;color:var(--mute)}
.big .v{font-size:40px;font-weight:700;letter-spacing:-.5px;color:var(--av1);font-variant-numeric:tabular-nums;line-height:1.1}
.big .u{font-size:13px;color:var(--mute)}
.chips{display:flex;flex-wrap:wrap;gap:8px}
.chip{
  font-size:12.5px;padding:4px 11px;border-radius:99px;
  border:1px solid var(--line);background:#0f1620;color:var(--mute);
  font-variant-numeric:tabular-nums;white-space:nowrap;
}
.chip.ok{color:var(--ok);border-color:rgba(61,220,151,.35);background:rgba(61,220,151,.08)}
.chip.warn{color:var(--warn);border-color:rgba(255,122,89,.35);background:rgba(255,122,89,.08)}
.chip.blue{color:var(--av1);border-color:rgba(77,163,255,.35);background:rgba(77,163,255,.08)}
.notes{margin:14px 0 0;padding:0;list-style:none;display:grid;gap:8px}
.notes li{
  font-size:13px;color:#b8c4d2;background:#10161e;border-left:2px solid var(--av1);
  padding:8px 12px;border-radius:0 8px 8px 0;
}
.notes li.w{border-left-color:var(--warn);color:#e0cfc9}
.notes li.g{border-left-color:var(--ok)}

/* ---------- 图表 ---------- */
svg{width:100%;height:auto;display:block;overflow:visible}
.legend{display:flex;flex-wrap:wrap;gap:16px;margin-top:10px;font-size:12px;color:var(--mute)}
.legend i{display:inline-block;width:18px;height:0;border-top-width:2.5px;border-top-style:solid;vertical-align:middle;margin-right:6px}

/* ---------- 字节表 ---------- */
table{width:100%;border-collapse:collapse;font-size:13px;font-variant-numeric:tabular-nums}
th,td{padding:9px 8px;text-align:right;border-bottom:1px solid #1d2632}
th:first-child,td:first-child{text-align:left}
thead th{color:var(--mute);font-weight:500;font-size:12px;border-bottom:1px solid var(--line)}
tbody tr:last-child td{border-bottom:none}
td.num{font-weight:600}
.tag{display:inline-block;font-size:11px;padding:1px 7px;border-radius:5px;background:#0f1620;border:1px solid var(--line);color:var(--mute)}

/* ---------- 预设 ---------- */
.presets{display:flex;flex-wrap:wrap;gap:9px}
.presets button{
  background:#0f1620;border:1px solid var(--line);color:#c6d3e2;border-radius:9px;
  padding:7px 13px;font-size:12.5px;cursor:pointer;transition:.15s;font-variant-numeric:tabular-nums;
}
.presets button:hover{border-color:var(--av1);color:#fff;background:#12213a;transform:translateY(-1px)}
.presets button b{color:var(--av1)}

/* ---------- min_qp 条 ---------- */
.mrow{display:grid;grid-template-columns:76px 1fr 132px;gap:12px;align-items:center;margin:9px 0;font-size:12.5px}
.mrow .bar{height:9px;border-radius:99px;background:#0b1016;border:1px solid var(--line);overflow:hidden}
.mrow .bar i{display:block;height:100%;border-radius:99px}
.mrow .val{text-align:right;color:var(--mute);font-variant-numeric:tabular-nums}
.mrow.hl .val{color:var(--warn)}

/* ---------- 详情 ---------- */
details{border:1px solid var(--line);border-radius:11px;background:#10161e;margin:10px 0;overflow:hidden}
details[open]{background:#0f151d}
summary{cursor:pointer;padding:11px 15px;font-size:13px;color:#b8c4d2;list-style:none}
summary::-webkit-details-marker{display:none}
summary::before{content:"▸ ";color:var(--av1)}
details[open] summary::before{content:"▾ "}
details .body{padding:0 15px 15px;font-size:13px;color:#a9b6c5}
details .body p{margin:8px 0}
details .body table{font-size:12px}
details .body th,details .body td{padding:6px}
code{
  background:#0a0f15;border:1px solid #1e2833;border-radius:5px;
  padding:1px 6px;font-size:12.5px;color:#9fd0ff;
  font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
}
.tbl-scroll{overflow-x:auto}

/* ---------- 规则 ---------- */
.rules{margin:0;padding-left:20px;font-size:13.5px;color:#b8c4d2}
.rules li{margin:9px 0}
.rules b{color:#fff}
footer{color:#55657a;font-size:12px;text-align:center;margin-top:26px;line-height:1.8}

@media (max-width:720px){
  .grid2{grid-template-columns:1fr;gap:4px}
  .swap{transform:rotate(90deg);padding:2px 0}
  .big .v{font-size:32px}
  .mrow{grid-template-columns:64px 1fr;gap:8px}
  .mrow .val{grid-column:2;text-align:right}
}
</style>
</head>
<body>
<div class="wrap">

  <h1>AV1 ⇄ H.264/HEVC <em>QP</em> 近似换算器</h1>
  <p class="sub">
    H.264 与 HEVC 共用 <b>0–51</b> 标度；AV1 是 <b>0–255</b> 的 qindex —— <strong>另一把尺子，禁止按数字直抄</strong>。
  </p>

  <!-- ========== 输入 ========== -->
  <section class="card">
    <div class="grid2">
      <div class="side h">
        <label><b>H.264 / HEVC QP</b><span>0–51</span></label>
        <div class="numrow">
          <input type="number" id="qpNum" min="0" max="51" step="0.5" value="38">
          <div class="pct" id="qpPct">—</div>
        </div>
        <input type="range" id="qpRange" min="0" max="51" step="0.5" value="38">
      </div>

      <div class="swap">⇄</div>

      <div class="side a">
        <label><b>AV1 qindex</b><span>0–255</span></label>
        <div class="numrow">
          <input type="number" id="qiNum" min="0" max="255" step="1" value="142">
          <div class="pct" id="qiPct">—</div>
        </div>
        <input type="range" id="qiRange" min="0" max="255" step="1" value="142">
      </div>
    </div>
  </section>

  <!-- ========== 结果 ========== -->
  <section class="card">
    <div class="result">
      <div class="big">
        <div>
          <div class="k">AV1 qindex ≈</div>
          <div><span class="v" id="bigQi">≈ 142</span> <span class="u">/ 255</span></div>
        </div>
      </div>
      <div class="chips" id="chips"></div>
    </div>
    <ul class="notes" id="notes"></ul>
  </section>

  <!-- ========== 曲线 ========== -->
  <section class="card">
    <h2>映射曲线：为什么“×5”只在高端近似成立</h2>
    <svg id="chart" viewBox="0 0 680 420" role="img" aria-label="H.264/HEVC QP 到 AV1 qindex 的映射曲线"></svg>
    <div class="legend">
      <span><i style="border-color:#4da3ff"></i>实测等价映射</span>
      <span><i style="border-color:#ff7a59;border-top-style:dashed"></i>×5 经验（错）</span>
      <span><i style="border-color:#5b6a7c;border-top-style:dotted"></i>恒等（数字直抄）</span>
      <span><i style="border-color:#ffb454;opacity:.8"></i>底表锚点</span>
    </div>
  </section>

  <!-- ========== 字节验证 ========== -->
  <section class="card">
    <h2>字节侧的现实核对（同片源 · rc=constqp · 同 preset · 720p · 平均 B/帧）</h2>
    <div class="tbl-scroll">
      <table>
        <thead>
          <tr>
            <th>内容类型</th>
            <th id="thH">H.264 @ QP</th>
            <th id="thV">HEVC @ QP</th>
            <th id="thA">AV1 @ qindex</th>
            <th>AV1 相对 H.264</th>
          </tr>
        </thead>
        <tbody id="byteBody"></tbody>
      </table>
    </div>
    <ul class="notes" style="margin-top:14px">
      <li class="g">
        在上表的<b>困难内容</b>上，换算器给出的 qindex 与 H.264 的帧大小基本对齐 —— 文档里的
        “字节等价”锚点表正是由这条曲线反推出来的。
      </li>
      <li class="w">
        切到<b>简单内容</b>看：同一对 QP / qindex 的字节数会出现几十个百分点的偏差。
        这说明低段映射<b>由内容决定</b>，任何换算公式都只是起点。
      </li>
    </ul>
  </section>

  <!-- ========== 快捷预设 ========== -->
  <section class="card">
    <h2>常用起点（点击套用）</h2>
    <div class="presets" id="presets"></div>
  </section>

  <!-- ========== min_qp 位置 ========== -->
  <section class="card">
    <h2>默认 min_qp 落在标度的什么位置</h2>
    <div class="mrow">
      <span class="tag" style="color:#ffb454;border-color:rgba(255,180,84,.35)">H.264</span>
      <div class="bar"><i style="width:37.3%;background:linear-gradient(90deg,#ff9c3a,#ffb454)"></i></div>
      <span class="val">19 / 51 = 37%</span>
    </div>
    <div class="mrow">
      <span class="tag" style="color:#b98cff;border-color:rgba(185,140,255,.35)">HEVC</span>
      <div class="bar"><i style="width:45.1%;background:linear-gradient(90deg,#8f5cff,#b98cff)"></i></div>
      <span class="val">23 / 51 = 45%</span>
    </div>
    <div class="mrow hl">
      <span class="tag" style="color:#4da3ff;border-color:rgba(77,163,255,.35)">AV1</span>
      <div class="bar"><i style="width:9%;background:linear-gradient(90deg,#2f7fd8,#4da3ff)"></i></div>
      <span class="val">23 / 255 = <b>9%</b></span>
    </div>
    <ul class="notes" style="margin-top:12px">
      <li class="w">
        “打开 <code>nvenc_enable_min_qp</code> 但不动数值” —— 在 H.264 上是 37% 的位置，
        在 AV1 上只有 <b>9%</b>，<b>几乎等于没设</b>。AV1 必须自己给数。
      </li>
    </ul>
  </section>

  <!-- ========== 明细 ========== -->
  <section class="card">
    <h2>依据与明细</h2>

    <details>
      <summary>锚点表（换算器的插值依据）</summary>
      <div class="body">
        <div class="tbl-scroll">
          <table>
            <thead>
              <tr><th>H.264/HEVC QP</th><th>28</th><th>34</th><th>38</th><th>42</th><th>47</th><th>51</th></tr>
            </thead>
            <tbody>
              <tr><td>字节等价的 AV1 qindex</td><td class="num">≈28</td><td class="num">≈100</td><td class="num">≈142</td><td class="num">≈178</td><td class="num">≈215</td><td class="num">≈242</td></tr>
              <tr><td>比值（AV1 : H.264）</td><td>1.0</td><td>2.9</td><td>3.7</td><td>4.2</td><td>4.6</td><td>4.7</td></tr>
            </tbody>
          </table>
        </div>
        <p>
          表间按<b>分段线性</b>插值。QP &lt; 28 的区间文档未给等价点，本工具按
          <b>恒等外推</b>处理，并显式标注“不可换算” —— 那一带的映射由内容决定，不是一把尺子。
        </p>
      </div>
    </details>

    <details>
      <summary>为什么 QP 不可按比例换算</summary>
      <div class="body">
        <p>
          QP 与量化步长是<b>指数</b>关系，约
          $$2^{\frac{QP-4}{6}}$$
          而 NVENC 的 AV1 qindex 并不是把 0–51 线性拉长：<b>低段步子细、高段步子粗</b>。
          所以“×5”这条常见经验只在接近 51 的高端近似成立。
        </p>
        <p>
          典型翻车：用 ×5 把 H.264 的 <code>34</code> 换算成 <code>170</code>，
          而实际等价值只有 <b>≈100</b> —— <b>上限会比你以为的松得多，超发更厉害</b>。
        </p>
      </div>
    </details>

    <details>
      <summary>原始实测数据：困难内容（逐帧随机噪声，最坏情况）</summary>
      <div class="body tbl-scroll">
        <table>
          <thead><tr><th>H.264 QP</th><th>19</th><th>23</th><th>28</th><th>34</th><th>38</th><th>42</th><th>47</th><th>51</th></tr></thead>
          <tbody>
            <tr><td>H.264 B/帧</td><td>45651</td><td>37555</td><td>28049</td><td>18765</td><td>16087</td><td>13320</td><td>10428</td><td>8215</td></tr>
            <tr><td>HEVC B/帧</td><td>27312</td><td>24979</td><td>21697</td><td>18135</td><td>15980</td><td>13061</td><td>10152</td><td>7959</td></tr>
          </tbody>
        </table>
        <table style="margin-top:10px">
          <thead><tr><th>AV1 qindex</th><th>23</th><th>51</th><th>76</th><th>92</th><th>102</th><th>128</th><th>138</th><th>152</th><th>176</th><th>204</th><th>230</th><th>255</th></tr></thead>
          <tbody>
            <tr><td>AV1 B/帧</td><td>29200</td><td>22784</td><td>20259</td><td>19192</td><td>18644</td><td>17120</td><td>16385</td><td>15356</td><td>13511</td><td>11359</td><td>9258</td><td>7270</td></tr>
          </tbody>
        </table>
      </div>
    </details>

    <details>
      <summary>原始实测数据：简单内容（渐变 + 移动方块）</summary>
      <div class="body tbl-scroll">
        <table>
          <thead><tr><th>H.264 QP</th><th>19</th><th>23</th><th>28</th><th>34</th><th>38</th><th>42</th><th>47</th><th>51</th></tr></thead>
          <tbody>
            <tr><td>H.264 B/帧</td><td>255</td><td>209</td><td>160</td><td>121</td><td>105</td><td>112</td><td>106</td><td>106</td></tr>
          </tbody>
        </table>
        <table style="margin-top:10px">
          <thead><tr><th>AV1 qindex</th><th>23</th><th>51</th><th>76</th><th>92</th><th>102</th><th>128</th><th>138</th><th>152</th><th>176</th><th>204</th><th>230</th><th>255</th></tr></thead>
          <tbody>
            <tr><td>AV1 B/帧</td><td>890</td><td>292</td><td>234</td><td>181</td><td>197</td><td>125</td><td>119</td><td>136</td><td>140</td><td>96</td><td>85</td><td>89</td></tr>
          </tbody>
        </table>
        <p style="margin-top:10px">
          注意 qindex 23 处 890 B/帧：<b>AV1 的可用下限段在简单内容上比 H.264 的 QP 19 还精细</b>
          （890 B vs 255 B）—— AV1 低段基本是留给近无损场景的。
        </p>
      </div>
    </details>

    <details open>
      <summary>实践规则（三条）</summary>
      <div class="body">
        <ol class="rules">
          <li><b>禁止换算，只认实测。</b>改 AV1 的 QP 上下限时，用 <code>qpmax=</code> 后缀确认已下发，
            再看帧大小 / 画质收敛。</li>
          <li><b>粗粒度起步值。</b>想要“H.264 38 的效果” → AV1 给 <b>≈140</b>；
            想要“H.264 47” → AV1 给 <b>≈215</b>。数字只能当起点。</li>
          <li><b>注意默认值位置。</b><code>min_qp_h264 = 19</code>（37%）、
            <code>min_qp_hevc = 23</code>（45%）、<code>min_qp_av1 = 23</code>（<b>9%</b>）。
            所以“打开开关但不动数值”在 AV1 上几乎等于没设。</li>
        </ol>
      </div>
    </details>
  </section>

  <footer>
    换算结果仅供起步参考 · 标定数据来自同片源 / rc=constqp / 同 preset / 720p 实测<br>
    低段（QP &lt; 28 或 qindex &lt; 28）映射由内容决定，不可作为依据
  </footer>
</div>

<script>
(function(){
  'use strict';

  /* ================= 数据 ================= */
  // 字节等价锚点（文档表）
  const ANCHORS = [
    {qp:28, q:28}, {qp:34, q:100}, {qp:38, q:142},
    {qp:42, q:178}, {qp:47, q:215}, {qp:51, q:242}
  ];
  // 低段按恒等补一个端点（文档未给等价点，仅供外推）
  const MAP = [{qp:0, q:0}].concat(ANCHORS);

  // 实测字节表 [QP 或 qindex, B/帧]
  const TB = {
    h264Hard:   [[19,45651],[23,37555],[28,28049],[34,18765],[38,16087],[42,13320],[47,10428],[51,8215]],
    h264Simple: [[19,255],[23,209],[28,160],[34,121],[38,105],[42,112],[47,106],[51,106]],
    hevcHard:   [[19,27312],[23,24979],[28,21697],[34,18135],[38,15980],[42,13061],[47,10152],[51,7959]],
    av1Hard:    [[23,29200],[51,22784],[76,20259],[92,19192],[102,18644],[128,17120],[138,16385],[152,15356],[176,13511],[204,11359],[230,9258],[255,7270]],
    av1Simple:  [[23,890],[51,292],[76,234],[92,181],[102,197],[128,125],[138,119],[152,136],[176,140],[204,96],[230,85],[255,89]]
  };

  const LOW_LIMIT = 28; // 低于此值：映射不可靠

  /* ================= 工具 ================= */
  const clamp = (v,a,b) => v<a?a:(v>b?b:v);
  const r1 = v => Math.round(v*10)/10;
  const $  = id => document.getElementById(id);

  // 分段线性：QP -> qindex
  function qpToQi(qp){
    qp = clamp(qp,0,51);
    for(let i=0;i<MAP.length-1;i++){
      const a=MAP[i], b=MAP[i+1];
      if(qp>=a.qp && qp<=b.qp){
        const t=(qp-a.qp)/(b.qp-a.qp);
        return a.q + t*(b.q-a.q);
      }
    }
    return 242;
  }
  // 分段线性反解：qindex -> QP
  function qiToQp(qi){
    qi = clamp(qi,0,255);
    if(qi <= MAP[0].q) return 0;
    for(let i=0;i<MAP.length-1;i++){
      const a=MAP[i], b=MAP[i+1];
      if(qi>=a.q && qi<=b.q){
        const t=(qi-a.q)/(b.q-a.q);
        return a.qp + t*(b.qp-a.qp);
      }
    }
    return 51; // qindex > 242：AV1 端比 H.264 51 还狠，无对应
  }

  // 对数空间插值（比特率曲线近似）
  function interpLog(tab, x){
    const n = tab.length;
    let a=null, b=null, t=0, extra=false;

    if(x <= tab[0][0]){
      a=tab[0]; b=tab[1];
      t=(x-a[0])/(b[0]-a[0]); extra = x < a[0];
    } else if(x >= tab[n-1][0]){
      a=tab[n-2]; b=tab[n-1];
      t=(x-a[0])/(b[0]-a[0]); extra = x > b[0];
    } else {
      for(let i=0;i<n-1;i++){
        if(x >= tab[i][0] && x <= tab[i+1][0]){
          a=tab[i]; b=tab[i+1]; t=(x-a[0])/(b[0]-a[0]); break;
        }
      }
    }
    if(!a){ a=tab[0]; b=tab[1]; t=0; }

    const v = Math.exp(Math.log(a[1]) + (Math.log(b[1])-Math.log(a[1]))*t);
    return { v: v, extra: extra };
  }

  const fmtB = v => (v==null || !isFinite(v)) ? '—' : Math.round(v).toLocaleString('en-US');

  /* ================= 状态 ================= */
  let qpVal = 38;
  let qiVal = qpToQi(38);

  function setQP(v){
    qpVal = clamp(v, 0, 51);
    qiVal = qpToQi(qpVal);
    render();
  }
  function setQI(v){
    qiVal = clamp(Math.round(v), 0, 255);
    qpVal = qiToQp(qiVal);
    render();
  }

  /* ================= 渲染 ================= */
  function render(){
    const qp = qpVal;
    const qi = qiVal;
    const qiR = Math.round(qi);

    // ---- 输入框回写（避开正在编辑的框）----
    const act = document.activeElement;
    if(act !== $('qpNum'))  $('qpNum').value  = r1(qp);
    if(act !== $('qiNum'))  $('qiNum').value  = qiR;
    $('qpRange').value = qp;
    $('qiRange').value = qiR;

    $('qpPct').textContent = (qp/51*100).toFixed(0) + '% of 0–51';
    $('qiPct').textContent = (qiR/255*100).toFixed(0) + '% of 0–255';

    // ---- 大数字 ----
    $('bigQi').textContent = '≈ ' + qiR;

    // ---- chips ----
    const chips = [];
    const notes = [];
    const lowSeg = qp < LOW_LIMIT;
    const ratio  = qp > 0 ? qi/qp : NaN;
    const x5     = Math.round(qp*5);
    const overPct = (qi>0 && qp>0) ? (x5/qi - 1)*100 : 0;

    if(lowSeg){
      chips.push({c:'warn', t:'低段 · 不可换算'});
    } else {
      chips.push({c:'ok', t:'比值 ' + ratio.toFixed(1) + ' : 1'});
    }
    if(qp > 0 && !lowSeg){
      chips.push({
        c:'warn',
        t:'×5 会写成 ' + x5 + '（高估 ' + Math.round(overPct) + '%）'
      });
    }
    if(qiR <= 23){
      chips.push({c:'blue', t:'qindex ≤ 23 · 近无损段'});
    }
    if(qiR >= 243){
      chips.push({c:'warn', t:'qindex > 242 · 比 H.264 QP 51 更狠'});
    }
    chips.push({c:'', t:'占 AV1 标度 ' + (qiR/255*100).toFixed(0) + '%'});

    $('chips').innerHTML = chips
      .map(c => '<span class="chip ' + c.c + '">' + c.t + '</span>')
      .join('');

    // ---- 说明 ----
    if(lowSeg){
      notes.push({c:'w', h:
        'H.264 QP &lt; 28 落在<b>没有等价值</b>的区间：AV1 的 qindex 23 在简单内容上已经是 '
        + '890 B/帧，比 H.264 QP 19 的 255 B/帧还精细 —— AV1 低段是留给近无损场景的。'
        + '这里按<b>恒等外推</b>给出数字，<b>只能当起点</b>，必须实测。'
      });
    } else {
      notes.push({c:'', h:
        '按实测锚点插值：H.264/HEVC QP <b>' + r1(qp) + '</b> ≈ AV1 qindex <b>' + qiR + '</b>。'
        + '给 AV1 填这个数，帧大小才会大致和 H.264 的 QP ' + r1(qp) + ' 对齐。'
      });
    }
    if(qp >= 28){
      notes.push({c:'w', h:
        '别用 ×5：把 QP ' + r1(qp) + ' 乘 5 得 <b>' + x5 + '</b>，而等价值只有 <b>' + qiR + '</b>，'
        + '高估了 <b>' + Math.round(overPct) + '%</b> —— 字面意思是上限设得太松，<b>超发更厉害</b>。'
      });
    }
    if(qiR <= 23){
      notes.push({c:'w', h:
        'qindex ≤ 23 是 AV1 的近无损预留段，日常码控几乎用不到；'
        + '默认 <code>min_qp_av1 = 23</code> 只占标度的 9%，等于没设。'
      });
    }
    notes.push({c:'g', h:
      '落地流程：先按此值下发 → 用 <code>qpmax=</code> 后缀确认已生效 → 再看帧大小 / 画质收敛。'
      + '数字只是起点，不是结论。'
    });

    $('notes').innerHTML = notes
      .map(n => '<li class="' + n.c + '">' + n.h + '</li>')
      .join('');

    // ---- 表格 & 图 ----
    renderBytes(qp, qi);
    $('chart').innerHTML = buildChart(qp, qi);
  }

  /* ================= 字节表 ================= */
  function renderBytes(qp, qi){
    const hH = interpLog(TB.h264Hard,   qp);
    const hS = interpLog(TB.h264Simple, qp);
    const vH = interpLog(TB.hevcHard,   qp);
    const aH = interpLog(TB.av1Hard,    qi);
    const aS = interpLog(TB.av1Simple,  qi);

    $('thH').textContent = 'H.264 @ QP ' + r1(qp);
    $('thV').textContent = 'HEVC @ QP ' + r1(qp);
    $('thA').textContent = 'AV1 @ q' + Math.round(qi);

    const rows = [
      { name:'困难内容', sub:'逐帧随机噪声（最坏情况）', h:hH, v:vH, a:aH },
      { name:'简单内容', sub:'渐变 + 移动方块',           h:hS, v:null, a:aS }
    ];

    $('byteBody').innerHTML = rows.map(function(r){
      let diffTxt = '—', cls = '';
      if(r.h && r.a && r.h.v > 0){
        const d = (r.a.v / r.h.v - 1) * 100;
        const sign = d >= 0 ? '+' : '';
        diffTxt = sign + d.toFixed(0) + '%';
        cls = Math.abs(d) <= 15 ? 'ok' : (Math.abs(d) <= 40 ? '' : 'warn');
      }
      const flag = b => b.extra ? ' <span class="tag">外推</span>' : '';
      return '<tr>'
        + '<td><b>' + r.name + '</b><br><span style="color:#5f6e80;font-size:11.5px">' + r.sub + '</span></td>'
        + '<td class="num">' + fmtB(r.h && r.h.v) + (r.h?flag(r.h):'') + '</td>'
        + '<td class="num">' + (r.v ? fmtB(r.v.v) + flag(r.v) : '<span style="color:#4a5765">未测</span>') + '</td>'
        + '<td class="num" style="color:#4da3ff">' + fmtB(r.a && r.a.v) + (r.a?flag(r.a):'') + '</td>'
        + '<td><span class="chip ' + cls + '" style="font-size:12px">' + diffTxt + '</span></td>'
        + '</tr>';
    }).join('');
  }

  /* ================= 图表 ================= */
  function buildChart(qp, qi){
    const W=680, H=420, L=62, R=22, T=26, B=54;
    const X = v => L + (v/51)*(W-L-R);
    const Y = v => H-B - (v/255)*(H-T-B);
    const P = (x,y) => x.toFixed(1) + ',' + y.toFixed(1);
    let g = '';

    // 低段阴影
    g += '<rect x="' + X(0).toFixed(1) + '" y="' + T + '" width="' + (X(LOW_LIMIT)-X(0)).toFixed(1)
       + '" height="' + (H-T-B) + '" fill="#ffb454" opacity="0.06"/>';
    g += '<text x="' + ((X(0)+X(LOW_LIMIT))/2).toFixed(1) + '" y="' + (T+17)
       + '" fill="#ffb454" opacity="0.75" font-size="11" text-anchor="middle">低段 · 不可换算</text>';

    // Y 网格（255 / 5 段）
    for(let v=0; v<=255; v+=51){
      const y = Y(v);
      g += '<line x1="'+L+'" y1="'+y.toFixed(1)+'" x2="'+(W-R)+'" y2="'+y.toFixed(1)
         + '" stroke="#26313d" stroke-width="1"' + (v===0?'':' stroke-dasharray="2 4"') + '/>';
      g += '<text x="'+(L-10)+'" y="'+(y+4).toFixed(1)
         + '" fill="#8b98a8" font-size="11" text-anchor="end">'+v+'</text>';
    }
    // X 网格
    for(let v=0; v<=51; v+=5){
      const x = X(v);
      g += '<line x1="'+x.toFixed(1)+'" y1="'+T+'" x2="'+x.toFixed(1)+'" y2="'+(H-B)
         + '" stroke="#26313d" stroke-width="1" stroke-dasharray="2 4"/>';
      if(v % 10 === 0){
        g += '<text x="'+x.toFixed(1)+'" y="'+(H-B+18)
           + '" fill="#8b98a8" font-size="11" text-anchor="middle">'+v+'</text>';
      }
    }

    // 轴标题
    g += '<text x="'+((L+W-R)/2).toFixed(1)+'" y="'+(H-14)
       + '" fill="#6d7c8d" font-size="12" text-anchor="middle">H.264 / HEVC QP</text>';
    g += '<text x="16" y="'+((T+H-B)/2).toFixed(1)+'" fill="#6d7c8d" font-size="12" text-anchor="middle"'
       + ' transform="rotate(-90 16 '+((T+H-B)/2).toFixed(1)+')">AV1 qindex</text>';

    // 恒等线（数字直抄）
    g += '<line x1="'+X(0).toFixed(1)+'" y1="'+Y(0).toFixed(1)+'" x2="'+X(51).toFixed(1)+'" y2="'+Y(51).toFixed(1)
       + '" stroke="#5b6a7c" stroke-width="1.6" stroke-dasharray="1 5"/>';

    // ×5 线
    g += '<line x1="'+X(0).toFixed(1)+'" y1="'+Y(0).toFixed(1)+'" x2="'+X(51).toFixed(1)+'" y2="'+Y(255).toFixed(1)
       + '" stroke="#ff7a59" stroke-width="1.8" stroke-dasharray="7 6" opacity="0.85"/>';
    g += '<text x="'+(X(51)-6).toFixed(1)+'" y="'+(Y(255)+18)
       + '" fill="#ff7a59" font-size="11" text-anchor="end" opacity="0.9">×5</text>';

    // 映射曲线
    let pts = [];
    for(let v=0; v<=51; v+=0.25){ pts.push(P(X(v), Y(qpToQi(v)))); }
    g += '<polyline points="'+pts.join(' ')+'" fill="none" stroke="#4da3ff" stroke-width="2.6"'
       + ' stroke-linejoin="round" stroke-linecap="round"/>';

    // 锚点
    ANCHORS.forEach(function(a){
      g += '<circle cx="'+X(a.qp).toFixed(1)+'" cy="'+Y(a.q).toFixed(1)
         + '" r="3.4" fill="#0e1116" stroke="#ffb454" stroke-width="2"/>';
    });

    // 当前点
    g += '<line x1="'+X(qp).toFixed(1)+'" y1="'+T+'" x2="'+X(qp).toFixed(1)+'" y2="'+(H-B)
       + '" stroke="#4da3ff" stroke-width="1" opacity="0.35"/>';
    g += '<line x1="'+L+'" y1="'+Y(qi).toFixed(1)+'" x2="'+(W-R)+'" y2="'+Y(qi).toFixed(1)
       + '" stroke="#4da3ff" stroke-width="1" opacity="0.35"/>';
    g += '<circle cx="'+X(qp).toFixed(1)+'" cy="'+Y(qi).toFixed(1)
       + '" r="6.5" fill="#4da3ff" stroke="#0e1116" stroke-width="2.5"/>';

    // 当前点标签
    const lx = clamp(X(qp) + 14, L, W - 150);
    const ly = clamp(Y(qi) - 14, T + 12, H - B - 6);
    g += '<rect x="'+lx.toFixed(1)+'" y="'+(ly-15).toFixed(1)+'" width="130" height="22" rx="6"'
       + ' fill="#0d1826" stroke="#274157"/>';
    g += '<text x="'+(lx+10).toFixed(1)+'" y="'+(ly).toFixed(1)
       + '" fill="#9fd0ff" font-size="12" font-variant-numeric="tabular-nums">QP '+r1(qp)+' → q'+Math.round(qi)+'</text>';

    return g;
  }

  /* ================= 事件 ================= */
  $('qpRange').addEventListener('input', function(e){ setQP(parseFloat(e.target.value)); });
  $('qiRange').addEventListener('input', function(e){ setQI(parseFloat(e.target.value)); });

  $('qpNum').addEventListener('input', function(e){
    const v = parseFloat(e.target.value);
    if(!isNaN(v)) setQP(v);
  });
  $('qiNum').addEventListener('input', function(e){
    const v = parseFloat(e.target.value);
    if(!isNaN(v)) setQI(v);
  });
  $('qpNum').addEventListener('blur', function(){ this.value = r1(qpVal); });
  $('qiNum').addEventListener('blur', function(){ this.value = Math.round(qiVal); });

  // 预设
  var PRESETS = ANCHORS.map(function(a){
    return { qp:a.qp, q:a.q };
  });
  $('presets').innerHTML = PRESETS.map(function(p){
    return '<button data-qp="'+p.qp+'">H.264 '+p.qp+' → <b>q≈'+p.q+'</b></button>';
  }).join('') + '<button data-qp="19">H.264 19 → <b>低段</b></button>';

  $('presets').addEventListener('click', function(e){
    var b = e.target.closest('button');
    if(!b) return;
    setQP(parseFloat(b.dataset.qp));
    window.scrollTo({ top:0, behavior:'smooth' });
  });

  // 启动
  setQP(38);
})();
</script>
</body>
</html>
```
### 换算机制与数据验证

这个换算器用分段线性插值，把H.264/HEVC的QP值映射到AV1的qindex，再用对数插值估算帧大小，让您看到“字节等价”的实际效果。

**双栏输入与实时联动**：左右两个输入框分别对应H.264/HEVC QP（0–51）和AV1 qindex（0–255），拖动滑块或直接输入数值，另一侧会自动换算，结果区同步显示比值、偏差提示和低段警告。

**曲线图揭示经验规则误区**：图表同时画出实测等价曲线、×5经验线和恒等线，让“×5只在高端近似成立”一眼可见。锚点用橙色圆点标出，当前点用蓝色十字线和圆点定位，方便对照。

**字节级交叉验证**：表格并排展示困难内容和简单内容下三种编码器的帧大小，并标注AV1相对H.264的偏差百分比，帮助您理解低段映射为何不可靠。

**内置预设与规则说明**：常用QP值一键套用，默认min_qp位置用进度条直观对比，展开区域提供原始实测数据表和三条实践规则，方便查阅。
---

**优化建议：** 换算器的映射锚点集中在JS的`ANCHORS`常量里，字节实测数据在`TB`对象中，您可以按自己的片源调整这些数值来微调计算精度。

---
## 🧑‍💻 User

帮我查询下网络说说为什么这个qp映射关系非线性，我们实验测出来的结果符合预期吗

---
## 🧑‍💻 User

帮我查询下网络说说为什么这个qp映射关系非线性？

---
## 😀 默认助手 (7)

我把能拿到的一手资料查了一圈（搜索引擎对这类词的返回基本是噪声，所以我直接去翻了编码器源码里的量化表），然后用你的数据做了反算验证。结论是：**非线性这件事你的实验测对了，但其中一句“为什么”的解释是反的，另外还有一个更值得担心的口径问题。**

---

## 一、先给结论

| 你的判断 | 核验结果 |
|---|---|
| AV1 的 qindex 与 QP 不可按数字换算 | ✅ 成立，且能定量：同一数字下 AV1 的量化步长细 **12 倍**（q38 ≈ 4.4 vs QP38 ≈ 52） |
| 锚点表（28/100/142/178/215/242） | ✅ 完全可复现，误差 ≤0.5 格 |
| 上限侧等价、下限侧不等价 | ✅ 用步长独立验证成立 |
| “低段步子细、高段步子粗” | ❌ **反了**，实际是“两端粗、中段最细” |
| 锚点表 = “同画质”等价 | ⚠️ **不是**，它是“同码率”等价，两者差 35 格 |

---

## 二、为什么必然非线性（三个层次，从根本到工程）

<details open>
<summary><b>1. QP / qindex 本身不是“质量刻度”，而是“步长序号”</b></summary>

H.264 与 HEVC 的量化步长是：

$$Q_{step}(QP) = 2^{\frac{QP-4}{6}}$$

即 **每 +1 级步长 ×1.122（+12.2%），每 +6 级翻倍**。所以 0–51 这 52 个数字其实覆盖了 $\log_2(224/0.625)=8.49$ 个倍频程。QP 是“步长的以 2 为底的对数”，不是线性的质量轴——这一层本身就决定了任何跨编码器的换算都不会是线性的。

</details>

<details open>
<summary><b>2. 核心：AV1 的 qindex 是一张手调查表，格子密度不均匀</b></summary>

AV1 的 qindex 0–255 通过 libaom 的 `ac_qlookup_QTX` 表映射到步长（8-bit，Q3 定点，实际步长 = 表值 / 8）：

- qindex 0 → `4` → 步长 **0.5**
- qindex 255 → `1828` → 步长 **228.5**
- 总跨度 $\log_2(228.5/0.5)=8.84$ 个倍频程

注意这个巧合：**两个标度的“射程”几乎完全一样**（8.49 vs 8.84 倍频程），端点也几乎重合（0.625↔0.5，224↔228.5）——但一个用 52 格走完，一个用 255 格走完。多出来的 5 倍分辨率**分配得极不均匀**：

| qindex 位置 | 表中相邻格 | 每格步长变化 | 1 格 ≈ 多少 QP | 1 QP ≈ 多少格 |
|---|---|---|---|---|
| 1–5 | 8→12 | 7.5–10.7% | 0.6–0.9 | 1.1–1.7 |
| 23–30 | 30→37 | 2.6–3.1% | 0.21–0.25 | 4.0–4.7 |
| 60 | 67 | **1.46%** | 0.12 | 8.4 |
| 80–96 | 87→104 | **1.1–1.3%（最细）** | 0.09–0.11 | 9–10.8 |
| 120–255 | 152→1828 | 1.9% | 0.16 | 6.3 |
| *H.264 每个 QP* | — | *12.2%（恒定）* | *1* | *1* |

对比之下 H.264 是**恒定 12.2%/格**。所以“1 个 qindex 等于多少 QP”在 q80 附近是 0.09，在 q25 附近是 0.23，**相差 2.5 倍**——这就是非线性的直接来源。

**注意这里修正了你原文的一个说法**：不是“低段步子细、高段步子粗”，而是 **qindex ≈ 60–110 最细（1.1–1.3%/格），低段（q<40）最粗（3–11%/格），高端回到 1.9%/格**——是个“中间凹陷”的鞍形。

</details>

<details open>
<summary><b>3. “×5”不是错的，它是“平均值”</b></summary>

平均换算比其实算得出来：

$$\frac{0.1667\ \text{oct/QP}}{8.84/255\ \text{oct/qindex}} = \frac{0.1667}{0.0347} \approx \mathbf{4.8}$$

所以 **×5 恰好是对的“全局平均”**。它的问题不在于错，而在于被当成**处处成立**——局部真实值在 **×3.8（低端）到 ×10.4（中段 q60–130）** 之间摆动。

你原文的结论“×5 只在高端近似成立”基本站得住（高端 ×6.2，误差 24%），但**理由要换**：不是因为“高段步子粗”，而是因为中段被拉长成了一个鼓包，只有两端才接近平均值。

</details>

<details>
<summary><b>4. 两把尺子的“原点”没对齐 → 累计比值 1.0→4.7</b></summary>

qindex/QP 这个“累计比值”从 28 处的 1.0 一路涨到 51 处的 4.7，本质是两个原因叠加：

- AV1 的前 100 格就把 4.8 个倍频程走完了（每倍频程 21 格），后 155 格却走了 4.0 个倍频程（每倍频程 38 格）；
- H.264 从头到尾匀速（每倍频程 6 格）。

于是“同码率点”在一条标度上匀匀地走、在另一条上忽快忽慢，任何**单一乘数**都必然在对的地方错、在错的地方对。

</details>

<details>
<summary><b>5. 工程层噪声：这三条决定了“理论换算”永远只是近似</b></summary>

- **NVENC 是闭源硬件**，它自己的 qindex→量化器表没有公开，不保证等于 libaom 的表；
- AV1 不是“一个 QP”：`base_q_idx` + `delta_q`（分段）+ `delta_lf` + min/max 夹逼 + 硬件内部 AQ；
- 硬件里 qindex 还会顺带牵动环路滤波强度、RDO 阈值、工具开关，所以“字节 vs qindex”曲线**不是纯量化器曲线**。

</details>

---

## 三、你的实测结果符合预期吗？逐条核验

<details open>
<summary><b>A. 自洽性：满分 ✓</b></summary>

我用你的 AV1 字节曲线做对数插值，反查“和 H.264 同字节数”的 qindex：

| H.264 QP | H.264 B/帧 | 反查 qindex | 你的锚点 | 偏差 |
|---|---|---|---|---|
| 28 | 28 049 | 27.5 | 28 | −0.5 |
| 34 | 18 765 | 99.8 | 100 | −0.2 |
| 38 | 16 087 | 142.0 | 142 | **−0.0** |
| 42 | 13 320 | 178.3 | 178 | +0.3 |
| 47 | 10 428 | 214.9 | 215 | −0.1 |
| 51 | 8 215 | 242.4 | 242 | +0.4 |

误差全部 ≤ 0.5 格。**这张表没有算错，它就是“困难内容 + 等码率”的忠实产物。** 这也说明你实验的重复精度是够的。

</details>

<details open>
<summary><b>B. 形状：符合 ✓</b></summary>

用你自己的数据算“1 个 H.264 QP 相当于多少个 qindex”（困难内容，瞬时口径）：

| qindex 区间 | 23–51 | 51–76 | 76–92 | 92–102 | 102–128 | 128–152 | 152–176 | 176–204 | 204–230 | 230–255 |
|---|---|---|---|---|---|---|---|---|---|---|
| ≈ 格/QP | **5.9** | 11.1 | 15.4 | **17.9** | 15.8 | 11.2 | 9.7 | 8.4 | 6.6 | **5.4** |

一个干净的 U 形：两端 5–6，中段 15–18。和 libaom 表的斜率形状（低端陡、q80 附近最平、高端回升）**同形**，只是幅度大约放大 1.5 倍（因为比特对步长的响应小于 1 次方）。这说明你的字节曲线确实是“由量化表驱动的”，机制对得上。

</details>

<details open>
<summary><b>C. 两端结论：用步长独立验证，站得住 ✓</b></summary>

| 比较 | AV1 步长 | H.264 步长 | 判定 |
|---|---|---|---|
| 最细端：q23 vs QP19 | 3.75 | 5.50 | AV1 **细 1.47×** → “下限侧不等价、AV1 更精细”成立 |
| 最粗端：q255 vs QP51 | 228.5 | 224.0 | AV1 **粗 1.02×** → “上限侧等价且略狠”成立 |
| 可用跨度 | q23–255 = 5.93 oct | QP19–51 = 5.35 oct | AV1 的可用射程确实更宽 0.6 个倍频程 |

你原文那两条“直接推论”，方向、量级都对，而且现在是两套独立证据（字节 + 步长）互相印证。

</details>

<details open>
<summary><b>D. 需要修正：口径问题（这条最重要）⚠️</b></summary>

你的锚点表是**等码率**等价。如果目的是给 `min_qp`/`max_qp` 定“质量边界”，应该用**等步长/等画质**口径。两者差得很远：

| QP | H.264 步长 | 等步长 → qindex | 等字节 → qindex | 差 |
|---|---|---|---|---|
| 28 | 16.0 | 108 | 27.5 | **−80.5** |
| 34 | 32.0 | 150 | 100 | −50.5 |
| 38 | 52.0 | **177** | **142** | −35.0 |
| 42 | 80.0 | 200 | 178 | −21.8 |
| 47 | 144.0 | 231 | 215 | −16.1 |
| 51 | 224.0 | 254 | 242 | −11.6 |

含义：**q142 的量化步长只有 QP38 的 1/1.87。** 拿 142 当“H.264 38 的画质”，实际给 AV1 定了一个**更细**的边界——`min_qp_av1 = 142` 会允许 AV1 比你以为的更精细，低码率目标下更容易超发。要做质量夹逼，起点应该往 **170–180** 那一段走，然后用实测收敛。

</details>

<details open>
<summary><b>E. 需要修正：NVENC ≠ libaom ⚠️</b></summary>

等步长点（q177）和你实测的等字节点（q142）差了 35 格，而且这个差值随码率单调收缩（QP28 处 −80 → QP51 处 −12）。折算成“同码率下 AV1 步长细几倍”：**3.66 → 2.29 → 1.87 → 1.51 → 1.36 → 1.26**。

这个趋势本身是合理的（AV1 越好越省），但**低码率端 3.7 倍的步长差**大到不可能纯由编码工具解释——更可能是 **NVENC 的 qindex 表和 libaom 的表本来就不是同一张**。

**实践结论：任何“查表 + 理论步长”的换算都不能替代实测。** 你原文规则 1“禁止换算，只认实测”不只是保守，是有据可查的必然。

</details>

<details open>
<summary><b>F. 测量卫生：简单内容那组不能用来锚定 ⚠️</b></summary>

用简单内容反查同一套锚点，结果全崩：

| QP | H.264 B/帧 | 反查 qindex | 你表里的值 | 偏差 |
|---|---|---|---|---|
| 28 | 160 | 113.9 | 28 | **+85.9** |
| 38 | 105 | 197.3 | 142 | +55.3 |
| 51 | 106 | 196.6 | 242 | −45.4 |

而且简单内容里 **H.264 QP38 = 105 B，QP42 = 112 B（非单调上升）**，AV1 在 q102 / q152 / q176 / q255 也有回翘。这说明低码率下量化器已经不是主导变量（码控下限、工具切换、帧内刷新等在起作用），这类点做锚点会污染整张表。

你只拿困难内容做锚点是**对的**，建议在文档里标明这个理由。

</details>

---

## 四、要把这张表做“硬”，还差四件事

1. **补一条质量轴。** 现在的表只能回答“多少 qindex 花掉同样的比特”。加 `ffmpeg -psnr` / `libvmaf` 再扫一遍，得到“等 PSNR / 等 VMAF”的锚点，那才是给 min_qp/max_qp 用的口径。现在这两套数差 35 格，不能混用。
2. **多点素材。** 噪声 / 体育 / 动画 / 屏录 / 渐变各一条，每类给一组锚点，实际用时取包络而不是单点。
3. **重复 + 排除 AQ。** 每个点跑 3 次取中位数；`rc=constqp` 下确认 spatial AQ、lookahead、multipass 都关掉，否则你扫的不是纯量化器。
4. **看分布而不只看均值。** 平均 B/帧 会掩盖 P95 尖峰；码控超发多数是尖峰造成的。

---

## 五、要不要我改一下那个换算器？

基于上面的核验，我可以把它升级成更诚实的两版：

- **口径开关**：`等码率（现在的锚点表）` / `等画质（步长估算，+35 格左右）`，并显式提示两者不能混用；
- **修正文案**：把“低段步子细、高段步子粗”换成“两端粗、q60–110 最细（每格仅 1.1–1.3%）”，并加一张“1 QP ≈ N 格”的 U 形局部比值曲线；
- **加一条“理论 vs 实测”对照**，把上面 D、E 两张表搬进去，让“为什么不建议用理论换算”一眼可见。

要的话我直接改。

---

**核查说明（供你复现）**：AV1 的量化表来自 `libaom` 的 `av1/common/quant_common.c`（`ac_qlookup_QTX` / `dc_qlookup_QTX`，8-bit Q3），我取的是 `rav1e` 内置的同源副本（`src/quantize/tables.rs`）。H.264 步长用 ITU-T H.264 Table 8-13（0.625 起，每 6 级翻倍），与常用近似式 $2^{(QP-4)/6}$ 在 QP38 处误差 2%。NVENC 的 qindex→量化器表**无法获取**（闭源硬件），所以所有关于“NVENC 具体落在哪”的推断都是间接的——这一点我在上面的 E 条里已经标出来了。
