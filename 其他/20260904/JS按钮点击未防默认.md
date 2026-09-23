# JS按钮点击未防默认

## 🧑‍💻 User

我这段 JS 代码有点问题，点击按钮后页面会刷新，但我明明用了 preventDefault() ，帮我看看哪里错了：

document.getElementById('btn').addEventListener('click', function() {
event.preventDefault();
console.log('clicked');
});别光说问题，直接给我改好的完整代码。”

---
## 🤖 Assistant

## 问题所在

你的 `event` 是**未定义的变量**。函数是匿名回调，并没有把事件对象传进来，所以这里的 `event` 只是一个"意外存在"的全局变量：

- 在严格模式下它是 `undefined` → `event.preventDefault()` 直接抛 `TypeError`，`preventDefault()` 根本没执行 → 页面照常刷新。
- 在非严格模式下，某些浏览器会把它解析成已废弃的 `window.event`（IE 时代的遗留）。就算侥幸能跑，也**极度不可靠**，换浏览器/换环境就失效。

一句话：**事件对象必须通过回调函数的参数显式接收**，你没有接收它。

---

## 改好的完整代码

```javascript
document.getElementById('btn').addEventListener('click', function (event) {
  event.preventDefault();
  console.log('clicked');
});
```

就加了一个参数 `event`（名字随意，习惯上常写 `e`）。这样浏览器会把事件对象自动传进来，`preventDefault()` 才能作用到真正的点击事件上。

---

## 顺带提醒

如果这个 `btn` 是 `<button>` 且位于 `<form>` 内，`preventDefault()` 能阻止表单提交刷新。但如果按钮本身没有默认行为（比如普通 `<div>` 绑 click），那真正让页面刷新的可能是别的原因——比如父级 `<form>` 提交、或 `<a href>` 跳转。

**验证小技巧**：在回调里加一行

```javascript
console.log(event.type, event.target);
```

如果能看到输出，说明事件对象拿到了；如果点击后控制台什么都没打印、页面直接刷新，那问题就不在这个 `preventDefault`，而在别处触发了导航/提交。
