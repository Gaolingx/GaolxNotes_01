# 实现map解析整数的陷阱

## 🧑‍💻 User

这行 JavaScript 输出什么？
```
console.log(["1", "2", "3"].map(parseInt))
```

---
## 😀 默认助手 (7)

输出结果是：

```javascript
[1, NaN, NaN]
```

## 为什么？

问题出在 `map` 回调的签名和 `parseInt` 的签名“意外地”配合在了一起。

- `map` 会向回调传入 **三个参数**：`(element, index, array)`
- `parseInt` 的签名是：`parseInt(string, radix)`，第二个参数是**进制（radix）**

于是实际调用变成了：

```javascript
parseInt("1", 0, ["1","2","3"])  // → 1
parseInt("2", 1, ["1","2","3"])  // → NaN
parseInt("3", 2, ["1","2","3"])  // → NaN
```

逐个拆解：

| 调用 | 说明 | 结果 |
|------|------|------|
| `parseInt("1", 0)` | `radix = 0` 被视为默认值，按十进制解析 | `1` |
| `parseInt("2", 1)` | `radix = 1` 非法（合法范围是 2–36） | `NaN` |
| `parseInt("3", 2)` | `radix = 2`，但 `"3"` 不是合法的二进制数字 | `NaN` |

## 正确写法

如果想让每个元素都按十进制解析，用一个箭头函数包一层即可：

```javascript
["1", "2", "3"].map(n => parseInt(n, 10))
// → [1, 2, 3]
```

或者直接使用 `Number`（`Number` 只接收一个参数，不受此问题影响）：

```javascript
["1", "2", "3"].map(Number)
// → [1, 2, 3]
```

> **要点**：把函数直接传给 `map` 时，要当心它是否会对额外的 `index` / `array` 参数产生“副作用”。经典受害者就是 `parseInt`。
