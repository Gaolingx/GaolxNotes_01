# Unity热更技术与HybridCLR实现

## 🧑‍💻 User

unity standalone中play模式中实时运行c#代码+hotreload如何实现？
一般是需要编译的，但是有两种实现方式，一种是直接生成IL，预定义的runtime直接call就行，另外一种是在内存中生成assembly, reload时一并清除
Editor模式下.net以jit模式运行，是可以动态添加修改IL和assembly的
甚至可能不需要reload，运行时动态加载或者直接内存中生成IL
后者的经典用法是xlua，你可以看下它的实现，xlua有个特点是它的lua调用C#并不是反射，也不需要提前生成代码，而是lua调用时发现没有wrapper就现场使用emit函数生成一个函数
但这种技巧只能在Editor下使用，发布后一般都是il2cpp，不能用jit，就只能反射或者预先生成wrapper代码了，但开发期可以使用，大大方便了开发者
不知道hybridclr是怎么做的，这个可以调研下

---
## 😀 默认助手 (7)

I'll research HybridCLR's implementation. Let me pull its official documentation and source.

Got the README. Now let me pull the detailed docs on the mechanism, DHE, and hot reload.

Let me get the full doc sitemap and the key principle pages.

Now I have the doc map. Let me pull the core mechanism pages.

Let me get the remaining key pages: runtime loading, DHE, hot reload tech, and quickstart.

I have the core HybridCLR facts. Let me verify a couple of adjacent claims (Unity Mono's assembly-unload limits and Editor hot-reload practices) before writing up.

调研完成。主要依据 HybridCLR 官方文档（9.x）+ 仓库 README，页面包括 `intro`、`principle/sourceinspect`、`basic/aotgeneric`、`basic/methodbridge`、`basic/runhotupdatecodes`、`business/differentialhybridexecution`、`business/hotfix`、`business/reload/*`。

## 结论先行

**HybridCLR 走的是和你描述的 xlua/emit 路线完全不同的第三条路：它不依赖宿主的 JIT，也不在运行时编译 C#，而是自带一个 IL 解释器，并把 il2cpp 从"纯 AOT 运行时"改造成"AOT + Interpreter 混合运行时"。**

一句话拆解你关心的四点：

| 你的疑问 | HybridCLR 的答案 |
|---|---|
| 运行时没有 JIT，怎么执行新代码？ | 不用 JIT。自带**寄存器解释器**执行热更 dll 的 IL。 |
| `Assembly.Load` 在 il2cpp 上不是不支持吗？ | 改了 il2cpp **元数据管理模块**，支持运行时动态注册元数据，从而让 `Assembly.Load(byte[])` 真正可用。 |
| 泛型/反射怎么办（xlua 类方案的死穴）？ | **补充元数据**（专利技术）+ il2cpp 泛型共享，让 AOT 泛型函数体也能被解释执行。 |
| 现场 emit wrapper 那种技巧能移植吗？ | 不能，它靠宿主 JIT；HybridCLR 改成**离线预生成"桥接函数"**（官方文档自己都说"跟 lua 的 wrapper 函数原理相似"）。 |

关键点：**热更 dll 仍然是离线编译出来的**（`HybridCLR/CompileDll`），运行时只是"加载 + 解释"。它不是"运行时免编译"，而是"运行时免 JIT"。

---

## 1. 先校准坐标：决定"能否运行时造代码"的是 runtime，不是 Editor/build

你把 JIT 等同于 Editor，这个理解在目标平台维度上需要修正：

| 环境 | 执行方式 | `Reflection.Emit` / `DynamicMethod` | 可卸载的 `AssemblyLoadContext` |
|---|---|---|---|
| Unity Editor（桌面） | Mono **JIT** | ✅ 可用 | ❌ Unity 的 Mono 基本没有可靠的 collectible ALC |
| Standalone（Win/Mac/Linux）**Mono 后端** | Mono **JIT** | ✅ 可用 | ❌ 同上 |
| il2cpp（iOS / Android / WebGL / 主机 / 桌面 il2cpp） | 纯 **AOT** | ❌ 无 | ❌ |

所以真正的分水岭是 **Mono 后端（有 JIT） vs IL2CPP 后端（无 JIT）**。Standalone 用 Mono 后端时一样能 JIT——不是"只有 Editor 才行"。而 HybridCLR 走的是 IL2CPP 后端（必须把 Scripting Backend 切成 IL2CPP）。

顺带一个残酷事实：**Unity 长期没有可用的"卸载程序集"能力**。这正是 HybridCLR 把"可以 100% 卸载程序集"当卖点的原因——你设想的"内存中生成 assembly、reload 时一并清除"，在 CoreCLR 上靠 collectible ALC 可行，但在 Unity 的 Mono 上基本落不了地。

---

## 2. 你列的两条路线，落到具体机制

- **(a) 直接生成 IL + 预定义 runtime 直接 call** → 本质是 `Reflection.Emit` / `DynamicMethod`，或手工 patch 方法指针。**xlua 现场 emit wrapper** 就属这条：lua 调用发现没有 wrapper，就 emit 一个。**硬前提：宿主有 JIT。**
- **(b) 内存中生成 assembly、reload 时整体清除** → 需要**可卸载 ALC**（CoreCLR 有；Unity Mono 基本没有）。Roslyn 运行时编译 + `Assembly.Load(byte[])` 属于这条，但**加载后无法卸载**，反复 reload 就是内存泄漏。

两者的共同前提都是 **JIT**。il2cpp 上 (a)(b) 双双失效，只剩你提到的**反射**或**预生成 wrapper**。HybridCLR 的创新就在于：它不试图在 il2cpp 上"恢复 JIT"，而是**换一套执行引擎**。

---

## 3. HybridCLR 的核心：给 il2cpp 装一个解释器

README 原话：

> HybridCLR 扩充了 il2cpp 运行时代码，使它由纯 AOT runtime 变成 **AOT + Interpreter 混合 runtime**，进而原生支持动态加载 assembly。

官方"源码结构"页列出的具体工作：

1. 一个 C++ 实现的 **dll/PE 元数据解析库**；
2. **改造元数据管理模块，实现元数据的动态注册**（对 il2cpp 源码只改几百行，多为插入 hook，例如在 `GlobalMetadata::GetStringFromIndex` 里判断 `IsInterpreterIndex` 后转去解释器的元数据模块）；
3. **IL 指令集 → 自定义寄存器指令集**的 compiler（`transform/Transform.cpp` 的 `HiTransform::Transform`）；
4. 一个**高效的寄存器解释器**（`interpreter/Interpreter_Execute.cpp` 的 `Interpreter::Execute`，本质是个大 switch）；
5. 大量 **instinct（内建/内联）函数**提升性能。

**第 2 点是关键**：il2cpp 原本只认"编译期静态写死的元数据"，所以 `Assembly.Load` 无从谈起。HybridCLR 通过动态注册 + hook，让运行时新增的 assembly/类型/方法/字符串能进入 il2cpp 的元数据体系，于是 `Assembly.Load(byte[])` 变成了一条正常路径（`basic/runhotupdatecodes` 明确写"直接调用 `Assembly.Load(byte[])` 即可"）。

---

## 4. 两块必须啃下的"硬骨头"

### 4.1 AOT 泛型 → 补充元数据（HybridCLR 专利，社区版也能用）

il2cpp 执行 `IL → C++` 时**丢失了泛型函数的原始 IL 函数体**。例如你 AOT 里只实例化过 `List<int>`，热更里 `new List<float>()` 就会抛 `MissingMethodException: AOT generic method not instantiated`——因为拿不出 `List<T>.Add` 的函数体。文档还指出**值类型无法泛型共享**（内存对齐 + arm64 ABI 传参差异），以及 `async` 状态机糖会隐含引用 AOT 泛型函数。

HybridCLR 的解法：用 `HybridCLR.RuntimeApi.LoadMetadataForAOTAssembly(dllBytes, HomologousImageMode.SuperSet)` **把缺失的 AOT 泛型函数体元数据补回来**，解释器就能执行它。代价是内存（约 3–4 倍 dll 大小、不可回收）。商业旗舰版另有无需补充元数据的 **full generic sharing**。

> 这正好回应了你的顾虑：xlua 类方案泛型只能靠预生成，而 HybridCLR 是让**泛型函数体本身可解释执行**，所以能近乎无限制地写泛型/继承/反射。

### 4.2 桥接函数（method bridge）—— 就是你熟悉的 lua wrapper

interpreter 与 AOT 之间要双向调用，但两者**参数传递/存储方式不同**（解释器参数在解释器栈上），所以**每种函数签名都要一个桥接函数**做搬运。官方原话：

> 桥接函数必须提前在 AOT 部分生成，**这点跟 lua 的 wrapper 函数原理相似。**

- 签名归一化（`sbyte/bool→i1`、`int→i4`、`float→r4`、`class/指针→u`、`struct→s{序号}`…），同签名的函数共享一个桥接函数；
- 通过 `HybridCLR/Generate/MethodBridge`（或 `Generate/All`）**离线预生成**；
- 因为 AOT 部分是固定的，**桥接函数集在打包时就确定了，后续任何热更都不会缺**——官方特别强调"不用担心热更上线后突然出现桥接函数缺失"。

所以你的判断是对的："发布后 il2cpp 上只能反射或预生成 wrapper"。HybridCLR 选了预生成，但把它系统化成"按签名去重的 bridge functions"。

---

## 5. HybridCLR 的"热更/热重载"到底有哪几档

<details>
<summary>展开：四档能力对照（社区版 / Hotfix / 热重载版 / DHE）</summary>

| 档位 | 机制 | 关键限制 | 授权 |
|---|---|---|---|
| **社区版** | `Assembly.Load(byte[])` 加载热更 dll → **解释执行** | **不能卸载**旧 assembly，重复加载=内存泄漏 | 免费 |
| **Hotfix 动态热修复** | 运行时按"类+方法签名"**替换函数体**，`RuntimeApi.HotfixAssembly/HotfixAssemblies` | 只能改**函数体**，不能改类型定义；每次修复都占内存；需关 inline（`MaxMethodInlineDepth=0`） | 商业 |
| **热重载版** | 运行中**完全卸载/重载** assembly，代码可任意变化 | 不能卸载被依赖者；MonoBehaviour/ScriptableObject/`[Serializable]` 有诸多字段限制；不支持泛型脚本、析构、与 DOTS 不兼容 | 商业 |
| **DHE 差分混合执行** | AOT dll 与最新热更 dll 做**离线差分**（dhao 文件），未改函数跑原生 AOT，变化/新增函数跑解释器 | 加载 DHE 代码前不能执行对应 AOT 程序集任何代码；不能用 `[InitializeOnLoadMethod]`、Script Execution Order | 旗舰版 |

</details>

三个要点：

1. **都不是"运行时编译 C#"**。源码 → dll 依旧是打包期/热更期的离线编译（菜单 `HybridCLR/CompileDll`）。
2. **卸载是真的难**：热重载版能卸掉 99% 以上元数据，但 MonoBehaviour/ScriptableObject 对应的 `Il2CppClass` 被引擎内部引用而**无法释放**——多次加载只会发生一次未释放，不会持续增长。这恰是"reload 时一并清除"最棘手的部分。
3. **DHE 是性能答案**：官方称未改动部分跑原生 AOT，相较纯解释版可提升 3–30 倍。dhao 文件是离线算好的"哪些类型/函数变了"的清单。

---

## 6. 回到你的场景：Editor / Standalone play 模式实时跑 C# + hot reload

**如果你的目标是"开发期加速"（只在 Editor 或 Standalone-Mono play 模式）**，其实**不需要 HybridCLR**——因为那里有 JIT：

- **运行时编译 C#**：用 Roslyn（`Microsoft.CodeAnalysis.CSharp`）把源码编到内存 → `Assembly.Load(byte[])`。能"跑起来新代码"，但**无法卸载**旧程序集。
- **改已加载方法体（真正"改完立刻生效"）**：用 `Mono.Cecil` / `MonoMod.RuntimeDetour` 在内存里重写方法 IL 或打 detour。这类正是 Unity Editor 生态里各种 "hot reload" 资产的做法（不重新加载 assembly，因此绕开了卸载问题）。
- **现场 emit wrapper**：`DynamicMethod` 等——就是 xlua 那条路。
- **配合** Unity 2020+ 的 `Enter Play Mode Options`（关 domain reload）减少迭代等待。

**如果你的目标是"发布后也能热更/热重载"**（目标平台是 il2cpp，尤其 iOS/WebGL/主机），才需要 HybridCLR 这类**自带解释器**的方案。

**一个关键澄清**：HybridCLR 在 Editor 下其实"没做什么特别的事"。官方 Quickstart 的 Editor 分支是这样写的——

```csharp
#if !UNITY_EDITOR
    Assembly hotUpdateAss = Assembly.Load(File.ReadAllBytes($"{Application.streamingAssetsPath}/HotUpdate.dll.bytes"));
#else
    // Editor 下 dll 已被 Unity 当普通程序集加载，直接查即可
    Assembly hotUpdateAss = AppDomain.CurrentDomain.GetAssemblies().First(a => a.GetName().Name == "HotUpdate");
#endif
```

因为 Editor 是 Mono，热更 dll 就是**普通程序集**，解释器根本没上场。**解释器只在 il2cpp 目标上才真正接管执行**。

---

## 7. 一张总览对照

| 方案 | 执行引擎 | 依赖 JIT？ | 运行时编译 C#？ | 可卸载 assembly？ | 覆盖平台 |
|---|---|---|---|---|---|
| xlua / emit wrapper | 宿主 JIT | ✅ 强依赖 | 否（现场 emit wrapper） | 否 | 仅 Editor / Mono 后端 |
| Roslyn 运行时编译 + `Assembly.Load` | 宿主 JIT | ✅ | ✅ | ❌（Unity Mono） | 仅 Editor / Mono 后端 |
| Cecil / MonoMod 改方法体 | 宿主 JIT | ✅ | 否（改 IL） | 不需要卸载 | 仅 Editor / Mono 后端 |
| **HybridCLR 社区版** | **自带解释器** | ❌ | 否（离线编 dll） | ❌ | 全平台（含 iOS/WebGL） |
| **HybridCLR DHE** | AOT + 解释器差分 | ❌ | 否 | 否 | 全平台，接近原生性能 |
| **HybridCLR 热重载版** | 自带解释器 | ❌ | 否 | ✅（99%+） | 全平台 |

---

### 想继续深挖的入口

- 解释器核心：`hybridclr/transform/Transform.cpp` 的 `HiTransform::Transform`、`hybridclr/interpreter/Interpreter_Execute.cpp` 的 `Interpreter::Execute`（官方说断点这两个函数就能跟完整条 "IL 转换 → 解释执行" 链路）。
- 三个仓库：`il2cpp_plus`（对 il2cpp 的改动）、`hybridclr`（解释器核心）、`com.code-philosophy.hybridclr`（Unity 包/工作流）。
- API：`HybridCLR.RuntimeApi.LoadMetadataForAOTAssembly`（补充元数据）、`RuntimeApi.HotfixAssembly/HotfixAssemblies`（热修复）、`RuntimeApi.SetRuntimeOption`。
- 版本：9.x 支持 Unity 2022.3 / 2023.2 / 6000.x；2019–2021 需用 8.15.0。

**一句话总结**：你把问题定位成"如何在无 JIT 的 il2cpp 上恢复 xlua 那种 emit 能力"，而 HybridCLR 的答案是"**不恢复 JIT，而是把 IL 解释器和动态元数据注册搬进 il2cpp 本身**"——用解释执行换来了全平台热更，再用桥接函数（预生成 wrapper）和补充元数据（泛型函数体）补齐与 AOT 互操作的两块短板，最后用 DHE / 热重载这些商业能力解决性能与卸载这两个解释器方案的固有痛点。

---
## 🧑‍💻 User

如果是做mod热重载assembly有什么好办法

---
## 😀 默认助手 (7)

Good follow-up — mod热重载和游戏本体热更的约束其实不一样。让我先核实几个关键事实（尤其 Unity 的 Mono 到底能不能卸载程序集、以及 .NET 侧的现代做法），再给方案。

Let me confirm one more Unity-specific option (ILRuntime) before writing up.

已核实关键事实（.NET 官方的 collectible `AssemblyLoadContext`/unloadability 文档、插件加载教程，以及 ILRuntime 的设计目标）。下面给一套完整方案。

## 先分清两件事：**能编译 ≠ 能卸载**

很多人把"热重载 assembly"和"运行时能跑新代码"混为一谈，其实这是两件独立的事：

- **编译/加载**：到处都能做。Roslyn 运行时编译、预编译 dll 后 `Assembly.Load`、Cecil 改 IL——都不难。
- **卸载**：这才是真正的难点，也是"热重载"和"热加载"的分界线。旧 assembly 的**类型元数据 + 所有实例 + 静态字段**必须全部回收，否则每 reload 一次就泄漏一块。

**能不能卸载，取决于 runtime，不取决于你怎么写代码。**

## 三个决定性变量

| 变量 | 影响 |
|---|---|
| **Runtime 能力** | 有没有 JIT？有没有可卸载的加载上下文？(.NET Core ✅ / Unity Mono ❌ / Unity il2cpp ❌) |
| **mod 可信吗** | 需要沙箱边界 → 决定了能不能用"同进程加载 C# 程序集"这条路 |
| **要不要真释放内存** | 开发期迭代（能接受泄漏） vs 玩家运行期更新版本（必须真卸载） |

> 微软官方原话：**"Untrusted code cannot be safely loaded into a trusted .NET process. To provide a security or reliability boundary, consider a technology provided by your OS or virtualization platform."** —— 如果 mod 不可信，同进程加载 C# 程序集这条路本身就不成立。

---

## 方案谱系总表

| 方案 | 真卸载 | 沙箱 | 平台 | 典型用途 |
|---|---|---|---|---|
| **collectible `AssemblyLoadContext`** | ✅（协作式） | ❌ | .NET Core / 非 Unity | .NET 插件系统正统方案 |
| **AppDomain** | ✅（强制） | 弱 | .NET Framework，**Unity 不可用** | 老框架 |
| **独立进程隔离** | ✅（杀进程） | ✅ 强 | 全平台 | 最稳、UGC |
| **内嵌 VM（Lua/JS/WASM）** | ✅（丢 VM/实例） | ✅ 强 | 全平台含 il2cpp | Unity mod 实际主流 |
| **纯 C# IL 解释器**（ILRuntime / HybridCLR 热重载版） | ✅ | 取决实现 | 全平台含 il2cpp | 想用 C# 写 mod |
| **IL patch / detour**（Harmony/MonoMod） | ❌（不释放） | ❌ | Mono 全平台；il2cpp 需 HarmonyX | 改别人游戏、dev 迭代 |
| **不卸载、接受泄漏** | ❌ | ❌ | 任何有 JIT 的 | 仅开发期 |

---

## 1. .NET Core / 非 Unity：collectible `AssemblyLoadContext`（首选）

这是唯一"正统"的程序集卸载方案。核心是 `isCollectible: true` 的 ALC + **协作式卸载**（区别于 AppDomain 的强制卸载）：

```csharp
class ModLoadContext : AssemblyLoadContext
{
    public ModLoadContext() : base(isCollectible: true) { }
    protected override Assembly? Load(AssemblyName name) => null; // 依赖走默认上下文
}

[MethodImpl(MethodImplOptions.NoInlining)] // 关键：防止栈槽/寄存器把 ALC 钉住
static void LoadAndUnload(string path, out WeakReference alcRef)
{
    var alc = new ModLoadContext();
    alcRef = new WeakReference(alc, trackResurrection: true);
    Assembly a = alc.LoadFromAssemblyPath(path);
    a.EntryPoint?.Invoke(null, new object?[]{ new string[]{ "hi" } });
    alc.Unload();                    // 只是"发起"卸载
}

// 触发并等待真正回收
for (int i = 0; alcRef.IsAlive && i < 10; i++) { GC.Collect(); GC.WaitForPendingFinalizers(); }
```

<details>
<summary>展开：协作式卸载的"钉住根"清单 —— 这几乎是 mod 热重载 bug 的全部来源</summary>

`Unload()` 返回不代表卸载完成。以下任何一条成立，ALC 就回收不了：

- **栈槽 / 寄存器**里的引用（方法局部变量，甚至 JIT 隐式产生的临时）—— 这就是为什么要 `NoInlining` + 把 load/unload 放进独立方法。
- **静态字段**（最隐蔽：引用类型的静态字段实际存在 runtime 的静态数组里）。
- **强 GC 句柄**（`GCHandleType.Normal/Pinned`）。
- 某线程**调用栈上还有** ALC 内的方法。
- 你自定义 ALC 子类上的**字段**指向了 context 内的程序集/类型/实例（卸载期间 runtime 持强句柄，必须清空这些字段）。
- 待执行的 `RegisteredWaitHandle` 回调指向 context 内方法。

调试：WinDbg/LLDB + SOS，`!dumpheap -type LoaderAllocator` 找地址，`!gcroot <addr>` 找根链。
</details>

**两个必踩的坑**：

1. **Type identity**：同一个类型名在两个 ALC 里是**两个不同的类型**。所以 mod 要实现的接口/契约必须放在**默认上下文**的共享 assembly 里，mod 不能再自带一份拷贝（对应 MS 教程里的 `<Private>false</Private>` / `ExcludeAssets=runtime`）。否则 `typeof(IMod).IsAssignableFrom(t)` 永远为 false。
2. **一个 ALC = 一个版本**。升级 mod 就是新建 ALC、加载新版、让旧版被回收。

**但——这套在 Unity 里基本用不了。** Unity 的运行时（Mono 与 il2cpp）不提供可用的 collectible ALC，也不支持创建额外 AppDomain 做脚本隔离。这正是 HybridCLR 把"可以 100% 卸载程序集"当卖点的原因。

---

## 2. Unity 的残酷现实 → 换思路，别硬刚"卸载"

Unity 里，"卸载 assembly"几乎被三件事堵死：

- **Mono 无 collectible ALC**（类型即便能解析到也没有卸载语义）；
- **il2cpp 无 JIT**，反射/emit 受限；
- **MonoBehaviour / ScriptableObject / `[Serializable]`**：Unity 引擎内部按"程序集全名"引用这些类型，序列化又缓存反射信息 → **这些类型的元数据无法释放**（HybridCLR 自己文档也承认这部分卸不掉，只能复用）。

所以 Unity 生态的成熟做法不是"卸载 assembly"，而是**绕开它**，走下面四条路之一。

---

## 3. 内嵌 VM / 沙箱（做 mod 最推荐）

Lua（xlua/tolua）、JS（Puerts/QuickJS）、WASM（Extism/wasmtime/wasm3）。

**为什么它天然适合 mod**：

- **可卸载**：丢掉 VM 实例 / script 上下文即可，不需要碰 .NET 的 ALC；
- **可沙箱**：API 面由你定义，恶意的 mod 只能调到你暴露的函数（这正是 HybridCLR 要单独做"访问控制策略"才能达到的效果）；
- **跨平台**：il2cpp/iOS/WebGL 通吃；
- **mod 免编译**：玩家写脚本就能改。

**WASM 特别值得考虑**：给了你"能力型"沙箱 + 跨语言（mod 可以用 Rust/C/C#/Go 写，编译成 wasm）+ 真正可卸载可隔离。有现成 C# SDK 的框架（如 **Extism**），以及 `wasmtime-dotnet`、`wasm3`。

**代价**：性能（解释/JIT 边缘）、以及**宿主 API 桥接**又是"wrapper"问题——和你上一篇聊的 xlua emit/bridge 是同一类东西。

---

## 4. 纯 C# IL 解释器（想用 C# 写 mod，且目标含 il2cpp）

如果 mod 作者就是要用 C#、又要跑在 iOS/WebGL 上，只能"自带 IL 解释器"：

| 方案 | 机制 | 卸载 | 特点 |
|---|---|---|---|
| **ILRuntime** | Mono.Cecil 解析 PE + 自研解释器 + **自己的 `AppDomain`** | ✅ 整体丢弃 AppDomain 即可 | 纯 C#、开源；需 **CLR 绑定/Adaptor/Delegate 注册**（codegen）；性能一般 |
| **HybridCLR 热重载版** | 自带寄存器解释器 + 动态元数据注册 | ✅ 100% 卸载（商业版） | 快、生态好、带**访问控制**做沙箱；**付费** |

> ILRuntime 的设计目标原文就是"给没有 JIT 的环境（如 iOS）跑 C#"，它自带 AppDomain 概念，所以能"重新加载一个全新的程序集"。这是 HybridCLR 之前 Unity 热更新/热重载的经典答案。

---

## 5. IL patch / detour（Unity mod 生态的"事实标准"）

**Harmony / HarmonyX**、**MonoMod.RuntimeDetour**。它们不改 assembly、不卸载，而是**在运行时替换方法的原生实现**（Mono 走 detour，il2cpp 走 HarmonyX 的方法钩子）。

- **热重载实现方式**：`UnpatchAll()` 撤掉旧补丁 → 用新版程序集重新 `PatchAll()`。**行为变了，但旧程序集仍驻留内存** → 反复 reload 会泄漏。
- 优点：能魔改别人已有的游戏（BepInEx / MelonLoader 就是这套）。
- 本质：这是**"热重补丁"，不是"热重载 assembly"**。

---

## 6. 进程隔离（最稳的"真卸载"）

把 mod 跑在**独立进程**，宿主通过 IPC / 共享内存 / 命名管道通信。

- **真卸载**：杀进程 = 100% 回收，没有"钉住根"问题；
- **天然沙箱 + 崩溃隔离**：mod 崩溃不拖垮主进程；可加 OS 级权限限制；
- 代价：**没有跨进程的对象共享和 GC 边界**，只能传值/序列化数据；调试更麻烦；延迟。

不适合"mod 要频繁操作游戏对象"的强耦合场景，但适合"逻辑/计算/AI"类 mod。**如果 mod 不可信，这几乎是唯一保底方案**（和微软的建议一致）。

---

## 无论选哪种，把"可卸载"做对的工程要点

<details>
<summary>展开：一套通用的 mod 架构约束</summary>

1. **稳定 ABI / 契约 assembly**：宿主定义 `IMod` / `IModContext`，放在**默认上下文**；mod 只依赖它，宿主**绝不直接引用 mod 类型**。宿主侧全程只用接口。
2. **一个 mod 一个隔离边界**（一个 ALC / 一个进程 / 一个 VM），才能单独卸载。
3. **显式生命周期**：`Initialize()` / `Shutdown()`——`Shutdown` 里必须**退订所有事件、停止协程/线程、清空静态字典和反射缓存**。这是卸载成功的前提。
4. **跨边界只传接口/值/句柄**，不要把 mod 类型的实例长期存在宿主（静态字段是头号杀手）。
5. **弱引用缓存反射信息**（`MethodInfo` 缓存最能钉住 ALC）。
6. **版本化契约**：`IMod, Version=N`，mod 声明它编译所针对的宿主 API 版本。
7. **卸载可观测性**：`WeakReference`/内存增长监控，别靠"感觉没泄漏"。
8. **Unity 特化地雷**：协程、`static` 事件、序列化缓存（LitJson/Newtonsoft）、挂在 prefab 上的脚本、UnityEngine 对象持有 mod 类型引用——任一存在就无法卸载。
</details>

---

## 按场景的推荐

| 你的场景 | 推荐 |
|---|---|
| 非 Unity 的 .NET 应用 / 服务端 | **collectible `AssemblyLoadContext`**（标准答案） |
| Unity 桌面 + Mono 后端 + mod 可信 + 只求开发期迭代 | Roslyn 运行时编译 / Harmony patch，**接受不卸载**（泄漏可忍） |
| Unity（任意后端，含 il2cpp）+ 要真卸载 | **内嵌 VM（WASM/Lua/JS）** 或 **独立进程**；坚持用 C# 写则 **ILRuntime** / **HybridCLR 热重载版** |
| Unity + 魔改已有游戏 | **Harmony/HarmonyX**（patch，不卸载） |
| mod 不可信（UGC） | **WASM / Lua / JS 沙箱** 或 **独立进程**——不要在可信进程里直接 `Assembly.Load` |
| 目标含 iOS / WebGL | 排除一切依赖 JIT 的路线，只能 VM / IL 解释器 / 进程隔离 |

---

## 一句话结论

- **单纯要"能跑新代码"**：`Assembly.Load` + Roslyn/Cecil 随处可做，成本很低——但这不是热重载。
- **要"真热重载 assembly"（能卸载）**：只有三种物理上可行的边界——**collectible ALC**（仅 .NET Core）、**内嵌解释器/VM**（ILRuntime、HybridCLR 热重载版、Lua/JS/WASM）、**进程隔离**。Unity 里第一种基本不可用，所以实际就是后两种。
- **做 mod 的最优工程组合**：**稳定接口 ABI（宿主侧）+ 每 mod 一个隔离边界 + 边界外零强引用**。在这个骨架上，隔离边界选 ALC / VM / 进程，取决于你的 runtime、信任模型和性能预算。
- **如果 mod 不可信**：把"沙箱"当成一等需求，直接上 **WASM/脚本 VM 或独立进程**，别指望在可信 .NET 进程里靠限制 API 来沙箱（.NET 的 CAS 已死）。
