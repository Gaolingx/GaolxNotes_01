# xlua与c#交互瓶颈解析

## 🧑‍💻 User

xlua和c#层交互瓶颈在哪里

---
## 😀 默认助手 (7)

# xLua 与 C# 层交互的瓶颈在哪

## 先说结论

调用开销的来源可分为三层：**一次跨语言调用实际上等于 1 次“原生↔托管”穿越、N 次 P/Invoke 和 M 次托管堆分配**。因此，开销来自三层叠加：

1. **穿透成本**：每次调用都要跨越 native/managed 边界，并且逐参数做 P/Invoke 读栈；
2. **封送与生命周期成本**：字符串要复制、值类型要装箱、对象要走引用表查表、双向 GC root 都产生额外成本；
3. **次数**：单次开销是常数级（$O(1)$），真正的瓶颈永远是“每帧调了多少次”。

在实际工程中，最常见的两个导致性能严重下降的原因，往往与 xLua 本身无关，而在于配置和写法：

- 没有走代码生成，退化成 `xlua.reflection`（`MethodInfo.Invoke` + 装箱），会慢 1~2 个数量级；
- 高频走属性、字段访问（`v.x`、`t.Count`），每次都是一次完整的跨语言调用。

---

## 一、先拆解一次调用链

以 Lua 侧 `obj:Method(1, "abc")` 为例，实际执行过程如下：

```
Lua VM
  └─ 找到 userdata 的 metatable __index ---> ① 一次跨语言交互（方法查找）
  └─ 调用注册进来的 C function (wrapper) ---> ② native → managed 穿越
       ├─ lua_type(L, 1) / lua_tointeger(L, 2) / lua_tostring(L, 3)  ---> ③ 每参数 1~2 次 P/Invoke
       ├─ lua_tostring 返回 IntPtr → Marshal.PtrToStringUTF8  ---> ④ 一次 string 分配
       ├─ translator.GetObject(idx, type)  ---> ⑤ Dictionary 查表 + 类型校验
       ├─ 反射重载决议（未生成代码时）
  └─ 真实方法执行
  └─ 返回值 push 回栈（每返回值一次 P/Invoke）+ lua_settop 清栈 ---> ⑥
  └─ 异常路径：C# 异常 → Lua error → longjmp → C# 再 catch 转 LuaException ---> ⑦ 栈展开
```

关键点：**参数个数越多，P/Invoke 次数线性增长**（$O(n)$），而不是一次调用就能搞定。所以“把 10 个参数合成 1 个结构体或一个接口”这类批量化和降维手段，收益非常直接。

---

## 二、主要瓶颈清单

| 瓶颈点 | 根因 | 量级参考 | 规避手段 |
|---|---|---|---|
| **反射回退** | 类型未加入 `[LuaCallCSharp]` / `[CSharpCallLua]`，走 `MethodInfo.Invoke`、`Activator.CreateInstance`、参数数组装箱 | 相比生成代码慢 **10~100 倍** | 补全生成列表，CI 里校验 `LuaCallCSharp` 覆盖 |
| **属性/字段访问** | 每次 `t.x` 都触发 `__index`，是一次完整跨语言调用 | 比 Lua 原生 table 访问慢 **几十倍** | Lua 侧用局部变量缓存；热点数据改成一次性拉取 |
| **string 往返** | `lua_tostring` → `PtrToStringUTF8` 必然分配；C# → Lua 也要复制 bytes | 每次 1 次分配 | 避免每帧传字符串；用 id/hash 代替；批量传 |
| **值类型装箱** | `LuaFunction.Call()` 返回 `object[]`，`LuaTable.Get<T>()` 泛型装箱 | 每次调用 N 次分配 | 用生成的强类型 delegate 接口替代 `Call`/`Get` |
| **对象引用表** | C# 对象 ↔ Lua userdata 靠 `translator.objects` / `reverseMap` 双向 Dictionary | 每次 1~2 次哈希 | 减少跨语言传对象，传 int id 或数组索引 |
| **Lua 对象包装** | Lua 的 table/function 传进 C# 会 `new LuaTable`/`LuaFunction`，并注册引用槽 | 每次 1 个托管对象 + 槽位 | 缓存复用，**绝不要每帧 new** |
| **委托/回调** | `Get<Action>(...)` 走 `DelegateBridge` 缓存查找，未命中要创建 delegate（IL2CPP 下更贵） | 未命中显著更慢 | 提前 `Get` 一次并缓存；AOT 预生成 |
| **重载决议** | 生成代码对所有候选重载逐一 `lua_type` 判断，if-else 链 | 重载数 × 参数数 | 避免大量重载；用不同方法名 |
| **`lua_pcall` 保护调用** | C# 调 Lua 一律走 pcall（setjmp），比 `lua_call` 慢 | 常数开销 | 无法避免，只能减少次数 |
| **异常路径** | C# 异常要包装成 Lua error 再 rethrow，涉及栈展开 | 极贵（微秒~毫秒级） | 热路径不要用异常做流程控制 |
| **双 GC** | Lua 增量 GC + C# GC 交互；跨引用导致对象无法回收 | 阶段性尖峰 | 控制 `LuaEnv.GC` 时机；避免跨 GC 强引用 |
| **`xlua.hotfix`** | 热更替换后每次调用多一层 Lua 侧查表/转发 | 叠加一次间接寻址 | 非必要不 hotfix 高频接口 |

---

<details>
<summary><b>三、为什么「属性访问」比「方法调用」更致命（展开）</b></summary>

方法调用至少是“一次调用拿到全部结果”。而属性访问是**一次调用只拿一个字段**：

```lua
-- 这个循环 = 6 次跨语言交互
for i = 1, n do
    local p = obj.position   -- 拿 struct
    local x, y, z = p.x, p.y, p.z
end
```

如果 Lua 侧每帧要读几十个字段，累计次数很容易上万。同理，`for k, v in pairs(csharpDict) do` 这种遍历 C# 容器的写法，几乎必然造成性能问题：它把 O(1) 的哈希遍历变成了 O(n) 次跨语言调用。

**正确做法**：让 C# 一次性把结果“推”给 Lua（返回数组或 table），而不是让 Lua 逐字段“拉”。
</details>

<details>
<summary><b>四、C# → Lua 方向的额外瓶颈（展开）</b></summary>

1. **必须 `lua_pcall`**：受保护调用，比 `lua_call` 多 setjmp 成本，且限制“不能跨 pcall 边界 yield”。
2. **`LuaFunction.Call()` 是“慢速通用口”**：`object[]` 入参 + `object[]` 返回，全部装箱。高频场景应改为 `[CSharpCallLua]` 标注的强类型 delegate。
3. **`LuaTable.Get<T>()` 的泛型解析**：内部仍要 push key → `lua_gettable` → 取栈值 → 转换。
4. **协程/`yield` 跨界**：涉及状态机与栈切换，比普通调用贵一个档次。
5. **`LuaEnv.Tick()`**：如果每帧手动 Tick，Lua GC step 会造成尖峰，注意别在战斗帧做全量 GC。
</details>

<details>
<summary><b>五、IL2CPP / AOT 平台的特殊情况（展开）</b></summary>

- 无法 `Reflection.Emit`，**所有 wrapper 必须预生成**；漏了就只能报错或退化。
- `Delegate.CreateDelegate`、泛型实例化受 AOT 限制，`[CSharpCallLua]` 标注必须完整，否则运行期 `ExecutionEngineException`。
- `[MonoPInvokeCallback]` 需要正确 AOT 标记，否则 wrapper 回调不可用。
- 好消息：IL2CPP 下 P/Invoke 是直接函数指针调用，通常**比 Mono 快**；Mono 的 marshaling stub 开销更明显。
- Lua 侧无 JIT（iOS 禁止），纯 Lua 逻辑本身是解释执行——**这时候瓶颈常常不在绑定层，而在 Lua 脚本自身的 table 访问和 GC**，不要误判。
</details>

---

## 六、定位手段（比猜更有效）

1. **数调用次数**：在生成的 wrapper 里加一行计数（或用 `Profiler.BeginSample` 包住），先拿到“每帧多少次跨语言调用”这个数字，它比任何微优化都关键。
2. **Unity Profiler 看 `LuaDLL.*` 和 `xlua.*` 采样**：区分“绑定开销”和“业务逻辑开销”。
3. **A/B 对比**：同一接口分别用生成代码 / 反射跑一遍，差异就是配置问题。
4. **xlua 自带的 Lua Profiler**：定位是 Lua 侧热点还是 C# 侧热点。
5. **盯分配**：`LuaFunction.Call` 的 `object[]`、`LuaTable` 的 new、string 封送——这些是 GC 尖峰的来源，用 Memory Profiler 看托管分配热点。

---

## 七、优化优先级（按收益排序）

1. **减少调用次数**：批量接口、数据推送代替拉取、把循环整体下沉到一侧（要么全在 Lua，要么全在 C#）。
2. **确保走生成代码**：`[LuaCallCSharp]`、`[CSharpCallLua]`、`[CSharpCallLua]` 标在 delegate 和 interface 上，CI 做覆盖校验。
3. **用强类型 delegate 替代 `LuaFunction`/`LuaTable` 通用口**。
4. **缓存一切跨语言创建物**：`LuaFunction`、`LuaTable`、delegate 提前 `Get` 并持有，杜绝每帧 new。
5. **热路径避免 string**：用 int/enum/hash 传参。
6. **避免异常做控制流**，热路径不抛异常。
7. **热点重构**：把每帧上千次的属性读写，改成一次性的 struct → Lua table 转换，或干脆在 C# 侧算完再推结果。

---

**一句话总结**：xLua 的瓶颈不是某个“神秘开销”，而是 **穿越次数 × 每次穿越的固定成本（P/Invoke、封送、分配、引用表）**。降低这个乘积的任何一项都有收益，而收益最大的是第一项：**让跨语言调用次数减少一个数量级，而不是让单次调用快 10%**。

---
## 🧑‍💻 User

确实，这种外挂脚本解释器瓶颈主要还是和native交互
怎么invoke尽量少反射尽量少gc优化alloc还有拆装箱啥的
unity那边的xlua<->c#基本也是围绕这些东西优化
能不能举几个例子讲解下

---
## 😀 默认助手 (7)

# xLua ↔ C# 优化的四类“税”，逐个举例拆开看

先给出一个成本模型，后面所有例子都可对应到它：

$$T_{\text{frame}} \;=\; \underbrace{N \cdot t_{\text{cross}}}_{\text{穿透税}} \;+\; \underbrace{\sum_{i=1}^{N} n_i \cdot t_{\text{pinvoke}}}_{\text{参数税}} \;+\; \underbrace{A \cdot t_{\text{alloc}}}_{\text{分配/GC 税}} \;+\; \underbrace{R \cdot t_{\text{reflect}}}_{\text{决议税}}$$

- $N$ = 每帧跨界调用次数（**唯一需要先量化的一项**）
- $n_i$ = 第 $i$ 次调用的参数/返回值个数
- $A$ = 每帧托管堆分配次数，$R$ = 走反射/动态决议的次数

下面按“最容易出问题、收益最大”的顺序举例子。

---

## 一、拆装箱：最隐蔽的一类

### 例 1：`LuaFunction.Call` 是最慢的通用通道

```csharp
// ❌ Lua 侧定义: function calc(a, b) return a + b, a * b end
LuaFunction f = luaTable.Get<LuaFunction>("calc");

void Update()
{
    object[] ret = f.Call(1, 2);        // 入参装箱 + 返回 object[] 新分配
    int sum = (int)ret[0];              // 拆箱
    int mul = (int)ret[1];              // 拆箱
}
```

实际发生的情况：

| 动作 | 代价 |
|---|---|
| `1` 和 `2` 作为 `object` 传入 | 2 次 `box int32` |
| 返回时 `object[]` | 1 次数组分配 + 2 次 box |
| `(int)ret[0]` | 拆箱（便宜但非零） |
| `Get<LuaFunction>` | **每个对象的一次构造包装对象并占用一个引用槽** |

每帧 60 次就是 **180 次堆分配**，GC 压力全都出在这里。

```csharp
// ✅ 强类型 delegate，零装箱零数组
[CSharpCallLua]
public delegate void CalcDelegate(int a, int b, out int sum, out int mul);

CalcDelegate _calc;   // Start 里 Get 一次，之后永久复用

void Awake() => _calc = luaTable.Get<CalcDelegate>("calc");

void Update() => _calc(1, 2, out int sum, out int mul);
```

> **要点**：xLua 生成代码时，会把 delegate 调用编译成“从 Lua 栈上直接 `lua_tointeger` 取值、直接调用给你”，完全绕过 `object`。这是收益最高的一条，几乎所有项目都能获得 10 倍以上收益。

<details>
<summary><b>例 2：`object` 参数是装箱的隐形入口（展开）</b></summary>

```csharp
// ❌ Lua 侧: emitter(1, "hit", {x=1})
[CSharpCallLua]
public delegate void EventHandler(object arg1, object arg2, object arg3);
```

只要参数类型是 `object`，**Lua 传进来的所有值类型都会被装箱**：Lua 里的 number 虽然原生是 double，但 `translator` 在给 `object` 槽位赋值时仍会产生 `double` 的装箱（如果走通用路径）。

```csharp
// ✅ 拆成具体类型的重载 / 用固定的 payload struct
[CSharpCallLua]
public delegate void IntEventHandler(int code, int param);

[CSharpCallLua]
public delegate void StrEventHandler(int code, string param);
```

**规律：热路径上 delegate 的签名里不要出现 `object`、`params object[]`，也不要用泛型 `T`，除非 `T` 被完全实例化。**

</details>

<details>
<summary><b>例 3：接口调用导致 struct 隐式装箱（展开）</b></summary>

```csharp
public interface IDamageable { void TakeDamage(int d); }

// ❌ 把 struct 当接口传
public struct Enemy : IDamageable { public void TakeDamage(int d) { } }

void Hit(IDamageable target) { target.TakeDamage(10); }  // Enemy 实例 → box

// ✅ 用 class，或让伤害走纯值类型参数
```

xLua 的引用表（`translator.objects`）遇到 box 后的 struct，还会**额外多一条强引用**，导致原本该被回收的值类型实例被钉住，形成“跨 GC 泄漏”：Lua 侧持有它，C# 侧回收不掉。

</details>

---

## 二、分配 / GC：真正的性能尖峰制造者

### 例 4：每帧 `Get<LuaFunction>` 等于每帧新建对象

```csharp
// ❌ 最常见的坏味道
void Update()
{
    luaTable.Get<LuaFunction>("onUpdate").Call();   // 每帧:new 包装对象 + 占引用槽 + 再 Call 分配 object[]
}
```

```csharp
// ✅ 缓存
private LuaFunction _onUpdate;
void Awake() => _onUpdate = luaTable.Get<LuaFunction>("onUpdate");
void OnDestroy() => _onUpdate.Dispose();
void Update() => _onUpdate.Call();

// ✅✅ 更好：把回调改成 Lua 反向调用 C# 注册的 delegate，或直接 hotfix 入口函数
```

**为什么重要**：`LuaTable` 和 `LuaFunction` 除了自身占托管堆，还会在 `LuaEnv` 的引用表里占一个 slot（用 freelist 分配）。如果不 Dispose，这些包装对象会像沙粒一样积累，最后 `LuaEnv.Dispose()` 时集中释放 → **退出场景卡顿**。

### 例 5：Lua 侧字符串 → C# 必然复制

```lua
-- ❌ 每帧拼字符串 + 复制到 C#
local name = "enemy_" .. i
obj:SetName(name)
```

```csharp
// C# 侧发生的事情:
// lua_tostring(L, n)  →  IntPtr
// Marshal.PtrToStringUTF8(ptr)  →  new string  ← 必分配
public void SetName(string n) { ... }
```

```lua
-- ✅ 传 id，让 C# 侧查表/拼接
obj:SetId(i)
```

```lua
-- ✅ 如果确实要传字符串：不要拼接，直接传常量 + 参数
obj:SetNameById(i)
```

**判断标准：每帧跨界的 string 超过个位数，就应当改成 id/enum/hash。** 顺带一提，Lua 侧用 `..` 拼字符串本身也是 Lua GC 的主要来源，这会导致双重分配。

### 例 6：闭包捕获 = 编译器偷偷生成 class

```csharp
// ❌ 每次 AddListener 都 new 一个编译器生成的闭包类 + 一个新 delegate
void Bind()
{
    button.onClick.AddListener(() => _luaTable.Get<Action>("onClick")());
}
```

```csharp
// ✅ 提前 Get 出来，方法组直接绑定（不捕获则无闭包分配）
private Action _onClick;
void Awake() => _onClick = _luaTable.Get<Action>("onClick");
void Bind() => button.onClick.AddListener(OnBtnClick);
void OnBtnClick() => _onClick();
```

### 例 7：`params object[]` 和可变参数

```csharp
// ❌ Lua 调用一个 params 方法
public void Log(params object[] args);   // 每次调用 1 次数组分配 + 全部装箱
```

```csharp
// ✅ 提供固定重载
public void Log(string msg);
public void Log(string fmt, int a);
public void Log(string fmt, int a, int b);
```

---

## 三、反射 / 决议：没配好会比上面所有加起来还慢

### 例 8：漏标 `[LuaCallCSharp]` → 直接退化成 `MethodInfo.Invoke`

```csharp
// ❌ 这个类没有出现在生成列表里
public class Player
{
    public void AddBuff(int id, float dur) { }
}
```

Lua 侧 `player:AddBuff(1, 2.0)` 走的是 `xlua.reflection` 兜底路径：

```
查找 MethodInfo (字典+缓存,首次贵)
  → new object[]{ box(1), box(2.0f) }      ← 数组 + 两次装箱
  → MethodInfo.Invoke                      ← 一个数量级
  → 异常被包成 TargetInvocationException  ← 再一层
  → 返回值再 box
```

实测上，**反射路径通常是生成代码的 10~100 倍**。而且这种退化完全静默，你在 Profiler 里只会看到 `MethodBase.Invoke` 占了一大片。

```csharp
// ✅ 加特性，或（推荐）在统一的地方静态注册
[LuaCallCSharp]
public class Player { ... }
```

```csharp
// ✅✅ 更可控：集中登记，防止漏网
public static class XLuaGenConfig
{
    [LuaCallCSharp]
    public static List<Type> LuaCallCSharp = new List<Type>
    {
        typeof(Player), typeof(Enemy), typeof(Transform),
        // ...
    };
}
```

> **工程化建议**：写一个 Editor 脚本扫描 `Assembly-CSharp`，把“被 Lua 引用过的类型”与生成列表比对；CI 中不通过就直接失败。这类问题的收益是数量级的，而成本只有一次性配置。

### 例 9：自定义 delegate / interface 漏标 `[CSharpCallLua]`

```csharp
// ❌ 没有特性
public delegate void BattleEvent(int code, int arg);
```

`luaTable.Get<BattleEvent>("cb")` 的行为：

- **Mono**：用动态方式兜底（`Expression`/Emit 造 delegate）——能跑，但每次 `Get` 都慢；
- **IL2CPP / iOS**：直接报错或运行期崩溃。

```csharp
// ✅
[CSharpCallLua]
public delegate void BattleEvent(int code, int arg);
```

接口同理：`Get<ILoader>()` 需要 xLua 在编译期生成一个 adapter 类；没有这个问题，就只能走动态代理（仅 Mono 可用）。

### 例 10：泛型方法是 AOT 的雷区

```csharp
// ❌ 泛型 + 值类型，AOT 下需要预实例化，且可能走反射
public T GetConfig<T>(string key) where T : class;

// Lua: cfg:GetConfig("x")  ← 无泛型信息，靠反射猜
```

```csharp
// ✅ 提供非泛型的强类型 API
public BattleConfig GetBattleConfig(int id);
public ItemConfig GetItemConfig(int id);
```

### 例 11：重载越多，决议链越长

xLua 生成的 wrapper 在参数数量不匹配时，会按 Lua 栈上的实际类型**逐个尝试候选重载**：

```csharp
// ❌ 7 个重载，每次调用都可能走 4~7 次 lua_type 判断
public void Set(int v);
public void Set(float v);
public void Set(string v);
public void Set(bool v);
public void Set(int a, int b);
public void Set(int a, float b);
public void Set(object v);   // ← 这个还会吞掉装箱
```

```csharp
// ✅ 拆名
public void SetInt(int v);
public void SetFloat(float v);
public void SetStr(string v);
```

---

## 四、调用次数：收益最大，也最容易被忽略

**单次调用怎么优化都是常数级的；次数是乘数。** 下面几个例子非常典型。

### 例 12：逐个属性“拉” → 改成一次性“推”

```lua
-- ❌ 每帧 = 1 + 1000×(属性访问次数) 次跨界
function update()
    for i = 1, #enemies do
        local e = enemies[i]
        DrawEnemy(e.id, e.x, e.y, e.hp)   -- 每次 e.x 都是一次跨界
    end
end
```

> 顺带说一个重要的细节：**xLua 把 C# 数组推给 Lua 是 userdata，`enemies[i]` 的每一次下标访问本身就是一次跨语言调用**。所以 `for i = 1, #arr` 遍历 C# 数组 = $O(n)$ 次跨界。

```csharp
// ✅ C# 侧一次性算好，推一个快照
int[] _ids = new int[MAX];
float[] _posHp = new float[MAX * 3];
int[] _countBuf = new int[1];

// 每帧调一次,把 buffer 填好
public void SnapshotEnemies(int[] ids, float[] posHp, int[] countOut);
```

```lua
-- ✅ 1 次跨界拿到全部，之后全在 Lua 本地 table 上跑
local ids, posHp, cnt = {}, {}, 0
function update()
    cnt = snapshot(ids, posHp)        -- 1 次跨界
    for i = 1, cnt do
        DrawEnemy(ids[i], posHp[i*3-2], posHp[i*3-1], posHp[i*3])  -- 全本地
    end
end
```

### 例 13：把循环整体移到一侧

```lua
-- ❌ 1000 次跨界（每次一个怪）
for i = 1, 1000 do
    enemy:Move(dt)
end
```

```lua
-- ✅ 1 次跨界
enemyMgr:MoveAll(dt)
```

同理，如果逻辑本身是纯数据计算，**整个循环要么全在 Lua，要么全在 C#**，不要在中间来回穿插。

### 例 14：`Vector3` 是 struct，语义是“拷贝”

```lua
-- ❌ 4 次跨界，而且 p 是拷贝，改了不回写
local p = transform.position     -- 跨界 1
p.x = 1                          -- 改的是 Lua 侧的拷贝
transform.position = p           -- 跨界 2
```

```lua
-- ✅ 用现成的整体 API,一次跨界
transform:SetPositionXYZ(1, 2, 3)   -- 或者 transform:Translate(...)
```

**规律：凡是 struct（`Vector3`/`Quaternion`/`Color`/`Rect`），在 Lua 里逐字段操作几乎肯定有问题，一律用 C# 侧封装好的整体方法。**

### 例 15：方法引用在循环外缓存

```lua
-- ❌ 每次 obj.Method 都触发 __index(跨语言一次) 再调用(再一次)
for i = 1, n do
    obj:Method(i)
end

-- ✅ 只查一次
local m = obj.Method
for i = 1, n do
    m(obj, i)
end
```

同样，`table.insert`、`math.floor` 这类全局，也建议在 Lua 侧先取 `local insert, floor = table.insert, math.floor` 制成局部缓存，也就是 Lua 层面的常规优化。

### 例 16：逐条事件回调 → 批处理

```csharp
// ❌ 1000 个伤害事件 → 1000 次 Lua 回调
public event Action<int, int, int> OnDamage;
```

```csharp
// ✅ 攒一帧,合并成一次
private int[] _codes = new int[1024], _a = new int[1024], _b = new int[1024];
private int _n;

void LateUpdate()
{
    if (_n > 0) { _onDamageBatch(_n, _codes, _a, _b); _n = 0; }
}
```

---

## 五、几张速查表

### 优化 ROI 排序（按“投入/收益”）

| 优先级 | 动作 | 典型收益 |
|---|---|---|
| ★★★★★ | 补全 `[LuaCallCSharp]` / `[CSharpCallLua]`,消灭反射路径 | 10~100× |
| ★★★★★ | delegate 强类型化,消灭 `LuaFunction.Call` 的 `object[]` | 5~20×,分配归零 |
| ★★★★☆ | 缓存所有跨语言创建物（`LuaFunction`/`LuaTable`/delegate） | 消除每帧分配 |
| ★★★★☆ | 属性/数组逐个访问 → 批量快照推送 | 10~1000×（视 n 而定） |
| ★★★☆☆ | 用 id 替代 string 跨界 | GC 尖峰显著下降 |
| ★★★☆☆ | 拆重载、避免 `params object[]` | 20~50% |
| ★★☆☆☆ | Lua 侧局部缓存、方法引用外提 | 10~30% |

### “看到这个写法就应当警惕”

```lua
obj:Method()          -- 循环里的每一次
arr[i]                -- C# 数组下标
v.x                   -- struct 字段
".." .. i             -- 拼字符串再跨界
for k,v in pairs(csDict) -- 遍历 C# 容器
luaFunc()             -- 每帧 new LuaFunction
```

---

<details>
<summary><b>六、怎么验证优化真的有效（而不是凭感觉）</b></summary>

1. **先数次数**：在 xLua 生成的 wrapper 入口插一行静态计数（或 `Profiler.BeginSample("XLua/Call")`），拿到“每帧跨界次数”这个基线数字。没有这个数字，所有优化都是盲猜。
2. **对比实验**：同一接口，一次用生成代码、一次强制反射，对比耗时差异 —— 这个差值直接告诉你配置是否完整。
3. **盯分配**：Unity Memory Profiler 看 `object[]`、`System.String`、`LuaTable`、`LuaFunction` 的每帧分配量，这三者归零基本就干净了。
4. **分帧看尖峰**：`LuaEnv.Tick()` 触发 Lua GC 的那一帧会凸起，用 `Profiler` 的帧视图确认是不是 GC 造成的。
5. **AOT 平台单独压测**：Mono 上跑得飞快的反射兜底路径，在 IL2CPP 上会直接失败或表现更差，**必须在目标平台验证**。

</details>

---

**总结**：Unity 侧围绕 xLua 优化，本质上就是把上面这四项成本压到最低——**砍次数（批量化、推代替拉）、砍穿透（强类型 delegate 替代通用口）、砍分配（缓存一切跨界物、用 id 替 string）、砍决议（生成列表补全、拆重载）**。其中前两项是数量级的，后两项是常数级的；代码可读性的代价由业务复杂度决定，但生成列表这一项没有任何理由不做。
