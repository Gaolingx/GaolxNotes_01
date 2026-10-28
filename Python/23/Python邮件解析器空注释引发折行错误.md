# Python邮件解析器空注释引发折行错误

## 🧑‍💻 User

在python标准库中发现了bug，但不知道值不值得报
```
from email.parser import Parser
from email import policy
m = 

Parser(policy=policy.default).parsestr("To: " + "(" * 39 + ")" * 39 + "\r\n\r\nx")

m.as_string()   # IndexError: list index out of range
```
是parser的bug，但是感觉除非刻意为之，没人会构造一个这样的输入。

---
## 😀 默认助手 (7)

## 结论先说

**这个 bug 值得修，但不值得由你另开一个 issue** —— 同类问题已经被人报了，现在正挂在 open 状态：

- issue `gh-159621` *"email: IndexError when folding address headers containing empty comments"*（3.12 / 3.13 / main 都能复现）
- PR `gh-159622` *"gh-159621: Handle empty lists in TokenList.startswith_fws"*，状态 `awaiting review`

你最好的做法是**把你的最小复现追加到 `gh-159621` 下面**（你的形态和他们报的不一样：他们是 `()`,你是嵌套括号），顺便在 PR 上 review 一句。

---

## 根因（已定位到具体一行）

`email/_header_value_parser.py:151`（3.13.2，3.12 同）：

```python
class TokenList(list):
    def startswith_fws(self):
        return self[0].startswith_fws()   # 空 list → IndexError
```

空注释 `()` 会被解析成一个**长度为 0 的 `Comment` TokenList**。折行时 `_refold_parse_tree()` 用 `if newline or part.startswith_fws():` 判断能否把这一段挪到新行，递归下潜到那个空节点就炸了。

我遍历了你那段输入生成的解析树，确实有这个空节点：

```
empty TokenLists in parse tree: [('Comment', 40, "'()'")]   # 深度 40，唯一的空 TokenList
```

所以你的输入和 issue 里那个 `"a"*50 + "@example.com,()b@example.com"` 是**同一个根因**，只是触发路径不同。

<details>
<summary><b>为什么阈值正好是 39 层括号（点击展开推导）</b></summary>

默认 `max_line_length = 78`，注释长度是 $2n$：

| $n$ | 注释长度 | 结果 |
|---|---|---|
| $\le 37$ | $2n \le 74$，加上 `To: ` 后 $\le 78$ | 整条头放得下，根本不折行 |
| $38$ | $2n+1 = 77 \le 78$ | 注释能独占一行（`To:\n ...`），此时 `newline` 是那个被抢走的空格（真值）→ `or` 短路，**不会**调 `startswith_fws` |
| $\ge 39$ | $2n+1 = 79 > 78$ | 连独占一行都放不下 → 只能拆成子 token 递归尝试 → 一路下潜到最里层的空 `Comment` → IndexError |

（这段是按 3.12 的折叠逻辑推的，和你的 39 吻合）
</details>

<details>
<summary><b>你那个版本 vs 当前版本：坑还在，只是要换形状</b></summary>

在 3.13.2 上跑你的原样代码**已经不抛异常**了（`To:` 折成两行），3.12→3.13 期间 `_refold_parse_tree` 被重写成非递归版本了。

但缺陷本身没修，换个形状照样炸（我在 3.13.2 实测）：

```python
from email.parser import Parser
from email import policy

m = Parser(policy=policy.default).parsestr(
    "To: " + "a" * 50 + "@example.com,()b@example.com\r\n\r\nx")
m["To"].defects        # ()  ← 解析阶段报不出任何 defect
m.as_string()          # IndexError: list index out of range
m.as_bytes()           # 同样 IndexError
str(m)                 # 同样 IndexError
```

traceback 末端就是 `startswith_fws` 自己递归了 6 层后踩空。

顺带：同一族输入在 $n \approx 500$ 以上会直接 `RecursionError`（`"("*500 + ")"*500` 只要 1 KB 的 header），这个 PR 也没管。
</details>

---

## 为什么"刻意为之没人会构造"这个理由不成立

1. **输入面完全是外部的。** 任何收信/读第三方邮件的程序——邮件网关、反垃圾、归档、自动转发、把邮件当模板渲染的服务——只要对不可信 header 调 `as_string()` / `str()` / `as_bytes()` 就暴露了。攻击成本是 82 字节的 header，不是我需要猜的什么奇异输入。
2. **抛的是 `IndexError`，不是 `email.errors.*Defect`。** 标准库对畸形头的契约是"记录 defect 继续跑"，直接抛异常是违约，而调用方几乎不会去 catch `IndexError`。
3. **炸在输出阶段。** `parsestr()` 成功、`defects == ()`，防御性检查**没有落点**——你没法在解析后先校验再折叠。
4. 修法成本是 1 行。

所以定性上：不是"故意构造所以不算"，而是 **untrusted input 上未捕获异常 / 轻量 DoS**，属于真 bug。不过它不涉及内存安全、也没绕过安全边界，走普通 bug 流程即可，不必套 SECURITY.md。

---

## 你现在可以做的三件事

<details open>
<summary><b>1. 去 gh-159621 补复现（不要另开）</b></summary>

评论里建议写清这几点，维护者会很高兴：

- 另一种形态：**嵌套空注释**，不只是 `(),`；
- 触发条件：header 总长跨过 78 列、进入折行路径才触发（用 `max_line_length` 克隆 policy 可以稳定复现，便于写进 test）；
- 版本差异：3.12 上你的 39 层原样代码就炸，3.13.2 上换成 `()` 形态才炸；
- 期望行为：应当记一个 `Defect` 并按原样输出，而不是抛异常。
</details>

<details>
<summary><b>2. 顺手 review PR gh-159622 的两个点</b></summary>

PR 的修法是 `return self and self[0].startswith_fws()`：

- 空列表时返回的是 `[]` 而不是 `False`，真值判断没问题但返回类型不一致，`return bool(self) and self[0].startswith_fws()` 更干净；
- 这个补丁只堵了 `IndexError`，没管深嵌套（$n \approx 500$ 起）的 `RecursionError`。同一条 issue 里提一句，可能就一起修了。
</details>

<details>
<summary><b>3. 打补丁前的临时 workaround</b></summary>

```python
import re
from email.parser import Parser
from email import policy

def safe_parse_header(raw: str):
    try:
        return Parser(policy=policy.default).parsestr(raw).as_string()
    except (IndexError, RecursionError):
        # 退回不折行的 policy，原始内容不会丢
        return Parser(policy=policy.compat32).parsestr(raw).as_string()
```

- `policy.compat32` 不做 refold，实测你这个输入正常输出；
- 或者入库前把注释整体剥掉 / 限制嵌套深度——顺便，RFC 5322 本来就要求单行 ≤ 998 字节，收信侧限长是最省事的兜底。
</details>

> 顺带一句：报的时候记得带上你的 `sys.version`。你原样代码在 3.13.2 上不炸，说明你在 3.12 或更早，这个信息对维护者判断影响面很有用。

---
## 🧑‍💻 User

https://github.com/python/cpython/pull/159622
复现结果是，只要长度超过policy的folding limit然后再放个empty comment就能触发

---
## 😀 默认助手 (7)

刚好，我按照你这条结论做了个矩阵测试，发现它**基本成立但有一个重要例外**。直接看数据：

现在精确复刻一下踩空的瞬间（打印 `_refold_parse_tree` 当时的局部状态），顺便把阈值卡准：

数据显示你的结论方向对，但**不是"超过 limit 就触发"**。先看这一行的算术细节：

模拟 PR 的一行修复后出现了一个**值得在 review 里提的问题**。先确认一下：

你的复现结论方向是对的，但**"超过 limit + 一个空注释"还差三个必要条件**。我把参数扫了一遍，精确化的结果是这样：

## 触发条件（4 条必须同时成立）

| # | 条件 | 反例（不触发） |
|---|---|---|
| 1 | **地址类头**（`To`/`Cc`/`Bcc`/`From`/`Reply-To`/`Sender`），折叠会下潜到 CFWS 子 token | `Subject`/`Comments`/`X-Foo` 无论多长都不炸（unstructured 不走这条分支） |
| 2 | 值**长到必须"在中间"折行**，即整段挪到续行也放不下（≥2 次折行） | `a*49@example.com,()b@…`（77 字符）能整段塞进续行 → OK |
| 3 | 需要折行的那个 token **以空注释开头** | `(())` 也行，只要是空的 |
| 4 | 它**紧贴**前一个非空白字符，**且后面还有 token** | `, ()b@…`（隔了个空格）→ OK；`a*50@…,()`（注释在行尾）→ OK |

实测数据（`max_line_length=78`，value = `"a"*k + "@example.com,()b@example.com"`）：

```
k=49  value_len=77 -> OK            # 整条能塞进续行，根本没有"折到一半"的状态
k=50  value_len=78 -> IndexError    # 必须拆中间 → 撞上空 Comment
k=90  value_len=118-> IndexError    # 之后全炸
```

条件 4 的证据，同一位置只差一个空格：

```python
"To: " + "a"*50 + "@example.com,()b@example.com"    # IndexError
"To: " + "a"*50 + "@example.com, ()b@example.com"   # OK
```

原因就在 `_header_value_parser.py:2898`：

```python
newline = _steal_trailing_WSP_if_exists(lines)
if newline or part.startswith_fws():     # ← 前面有空格时 newline 为真，短路躲过
```

我 dump 了踩空瞬间的现场，`part` 就是那个含空注释的 token，`lines[-1]` 以 `,` 结尾（偷不到空格）：

```
part='Address' str='()b@example.com'  lines[-1]='To: aaa…@example.com,'  maxlen=78
```

## 可以贴进 issue 的最小复现

不依赖默认 78 列，把 limit 压小后只要两个字符级的值：

```python
# 最干净的一个（无 defect）
pol = policy.default.clone(max_line_length=24)
Parser(policy=pol).parsestr("To: " + "a"*20 + "@b.c,()\r\n\r\nx").as_string()
# IndexError

# 极限压缩版
pol = policy.default.clone(max_line_length=6)
Parser(policy=pol).parsestr("To: aa,()x\r\n\r\nx").as_string()
# IndexError
```

---

## ⚠️ 我模拟了 PR 的那一行补丁，有两个 review 点你可以直接提

```python
return self and self[0].startswith_fws()   # PR 的修法
```

<details open>
<summary><b>问题 1：异常没了，但 <code>()</code> 从输出里被静默吞掉</b></summary>

打上补丁后实测：

```
ml=24  'a*20@b.c,()'        -> 'To: aaaaaaaaaaaaaaaaaaaa@b.c,'        ← () 没了
ml=40  'a*50@…,()b@example.com' -> 'To: aaa…@example.com,b@example.com'  ← () 没了
ml=78  同上                  -> '…,()b@example.com'                     ← () 还在
```

机制：`Comment` 的 `(` `)` **不是子节点**，是在 `__str__` 里拼出来的：

```python
class Comment(WhiteSpaceTokenList):
    def __str__(self):
        return ''.join(sum([["("], [self.quote(x) for x in self], [")"]], []))
```

所以空注释节点 `len() == 0` 但 `str() == "()"`。补丁让它 `startswith_fws()` 返回假 → fold 走到「不是 Terminal，拆子节点」分支 → `list(part)` 是空 → `()` 直接消失。

语义上无害（RFC 5322 里注释是透明的，`m['To'].addresses` 前后一致），但这是**输出保真度变化**，PR 的测试应该把期望输出明确钉下来，而不是只断言"不抛异常"。
</details>

<details>
<summary><b>问题 2：深嵌套仍然 RecursionError，补丁挡不住</b></summary>

`startswith_fws()` 是逐层递归的，注释嵌套约 500 层就爆：

```
n=500  未修=RecursionError   打补丁后=RecursionError
n=1500 未修=RecursionError   打补丁后=RecursionError
```

（`"("*500 + ")"*500` 只有 1004 字节）同一个 issue 里一并提，可能就不需要第二个 issue 了。
</details>

<details>
<summary><b>更彻底的修法（如果你愿意提 PR 建议）</b></summary>

与其在 `startswith_fws` 里判空，不如让 `_refold_parse_tree` 别把「子节点为空的 TokenList」当可拆对象——拆了等于丢内容：

```python
if not hasattr(part, 'encode'):
    newparts = list(part)
    if not newparts:          # 空 TokenList 没有可拆的东西，tstr 才是它的全部
        lines[-1] += tstr
        continue
    ...
```

或者从结构上治本：让 `Comment` 的 `(`/`)` 成为真实子节点（`parse_comment` 里建 ValueTerminal），这样 `len()`、`str()`、`startswith_fws()`、fold 四者自洽，`()` 也不会丢。这个改动大一些，但能一次消掉这一类"str() 有内容、子节点为空"的隐性坑。
</details>

## 一句可以直接发的评论

> 补充另一种触发形态（不止 `(),`）：**嵌套空注释**也行，`To: (×39 … ×39)` 在 3.12 上就以 78 字符触发同一处 `startswith_fws`。
>
> 而且触发条件比"超过 folding limit"更窄，需要同时满足：① 地址类头；② 值长到必须在中间折行（整段进不了续行，`k≤49` 不炸）；③ 折行点落在空注释上；④ 前一字符非空白（`_steal_trailing_WSP_if_exists` 返回空时才会真的调 `startswith_fws`）。用 `policy.default.clone(max_line_length=24)` + `To: a*20@b.c,()` 可以稳定复现，也能让回归测试不依赖 78 列。
>
> 另外两点：一是打上本 PR 的补丁后 `()` 会从输出中被静默丢弃（`Comment` 的括号不是子节点），建议测试里断言期望输出；二是 `startswith_fws` 的递归在注释嵌套约 500 层时仍会 `RecursionError`。
