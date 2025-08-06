# SVG实现Loading动画方法详解

### 🧑‍💻 User

如何用svg实现一个loading动画？

---

### 🤖 Assistant

下面我将介绍两种使用SVG实现Loading动画的方法：**旋转圆环动画** 和 **动态描边动画**。两种方法都只需少量代码，且高度可定制。

---

### 方法1：旋转圆环动画（简单实用）
```html
<svg class="spinner" width="60" height="60" viewBox="0 0 100 100">
  <circle cx="50" cy="50" r="40" fill="none" stroke="#3498db" stroke-width="10" 
          stroke-dasharray="150 100" stroke-linecap="round"/>
</svg>

<style>
.spinner {
  animation: rotate 1.5s linear infinite;
}

@keyframes rotate {
  0%   { transform: rotate(0deg); }
  100% { transform: rotate(360deg); }
}
</style>
```

**效果说明**：  
- 蓝色圆环围绕中心持续旋转
- `stroke-dasharray="150 100"` 控制虚线样式（150为实线长，100为间隔）
- `stroke-linecap="round"` 使线端变圆润

**自定义指南**：
1. 修改颜色：更改 `stroke` 属性值（如 `stroke="#e74c3c"`）
2. 调整粗细：修改 `stroke-width` 值
3. 调整速度：修改 `animation` 中的时间（如 `2s` 更慢）
4. 调整大小：修改 SVG 的 `width` 和 `height`

---

### 方法2：动态描边动画（更流畅）
```html
<svg class="loader" width="60" height="60" viewBox="0 0 100 100">
  <circle cx="50" cy="50" r="40" fill="none" stroke="#f3f3f3" stroke-width="10"/>
  <circle cx="50" cy="50" r="40" fill="none" stroke="#3498db" stroke-width="10" 
          stroke-dasharray="250" stroke-dashoffset="250" stroke-linecap="round"/>
</svg>

<style>
.loader circle:last-child {
  animation: dash 1.5s ease-in-out infinite;
}

@keyframes dash {
  0% {
    stroke-dashoffset: 250;
  }
  50% {
    stroke-dashoffset: 0;
    transform: rotate(135deg); /* 添加旋转增强效果 */
  }
  100% {
    stroke-dashoffset: -250;
    transform: rotate(450deg);
  }
}
</style>
```

**效果说明**：
- 蓝色弧线沿路径动态生长并旋转
- 灰色背景圆作为参照
- 使用 `stroke-dashoffset` 实现描边动画

**自定义指南**：
1. 弧线长度：修改 `stroke-dasharray`（值≈圆周长：2πr，r=40时≈250）
2. 颜色方案：修改两个圆的 `stroke` 值
3. 动画曲线：修改 `animation-timing-function`（如 `cubic-bezier(0.4, 0, 0.2, 1)`）

---

### 进阶技巧
1. **添加透明度变化**：
   ```css
   @keyframes dash {
     0%   { opacity: 0.3; }
     50%  { opacity: 1; }
     100% { opacity: 0.3; }
   }
   ```

2. **多色渐变**：
   ```html
   <defs>
     <linearGradient id="grad" x1="0%" y1="0%" x2="100%" y2="0%">
       <stop offset="0%" stop-color="#3498db"/>
       <stop offset="100%" stop-color="#e74c3c"/>
     </linearGradient>
   </defs>
   <circle ... stroke="url(#grad)"/>
   ```

3. **弹性效果**：
   ```css
   @keyframes dash {
     0%   { stroke-dashoffset: 250; }
     70%  { stroke-dashoffset: 0; }
     100% { stroke-dashoffset: -250; }
   }
   ```

---

### 原理说明
- **stroke-dasharray**：控制虚线模式（实线长+间隔长）
- **stroke-dashoffset**：控制虚线起始偏移
- **关键帧动画**：通过动态改变偏移量实现描边动画
- **旋转变换**：配合 `transform: rotate()` 增强动感

这些动画兼容现代浏览器（Chrome/Firefox/Safari/Edge），可通过调整参数轻松适配不同场景需求。