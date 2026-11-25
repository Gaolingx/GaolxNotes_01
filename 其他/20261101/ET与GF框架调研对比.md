# ET与GF框架调研对比

## 🧑‍💻 User

话说unity游戏框架是不是以必须用反射去实现core层东西（比如资源管理、初始化、网络服务等）
主要是反射拿来玩di很有意思 可以实现0侵入框架 还能hook一些引擎行为
unity项目经常后面做着做着框架变成大史山
把所有call换成attribute反射 在统一的地方注册就好看多了
反正ms自己的框架也这么玩（你去看efcore里边全是反射乱飞
反正维护交给ai
只是别玩字符串反射就行 那个维护起来比较地狱

---
## 😀 默认助手 (7)

## 先给结论

反射不是必须的。Core 层真正的需求只有三个：**解耦、生命周期管理、装配入口唯一**。反射只是达成这三点的手段之一，而且在 Unity 下是最贵、最不划算的那一种手段。

不过你的直觉里有一部分是对的：**Unity 里确实有一块地方反射近乎必需**，但原因不是框架设计，而是引擎抢走了对象构造权。这块先放一放，后面讲。

---

## 几个我觉得需要提醒的点

### 1. “0 侵入”在 Unity 里其实是个残缺版

MonoBehaviour 的 `new` 是 Unity 干的，不是你干的。所以不管你用 VContainer 还是 Zenject，MonoBehaviour 想拿依赖只能：

- 字段注入（反射或 `[Inject]`，一样是侵入，只是侵入到了字段上）
- `FindObjectOfType` / 序列化引用（更糟）
- **根本不让 MonoBehaviour 拿依赖** —— 它是 View/Adapter，逻辑放纯 C# 类

第三条才是真正 0 侵入的解法，而且它不需要反射。反射在这里看起来必需，是因为你默认了“MonoBehaviour 得被 DI 管”这个前提。

### 2. IL2CPP + 裁剪下，“反射魔法”是静默失效的重灾区

这条比“字符串反射难维护”严重得多：

- `[Preserve]` / `link.xml` 规则会膨胀成第二个隐式依赖网络，而且它和代码是**两份脱钩的真相**
- `Managed Stripping Level = High` 时，没被显式引用的 attribute 目标直接消失，注册表悄悄少几个条目
- iOS 上 `Expression.Compile` 没有 JIT，`MakeGenericType` 碰上值类型泛型直接抛异常
- `Assembly.GetTypes()` 在 Android 低端机上启动扫一遍，几十上百毫秒起

这些错误**编译期不报、Editor 下不报、只有真机上崩**。这是我最不喜欢反射 DI 的原因，不是性能。

### 3. “注册在统一地方就好看了” —— 好看的是注册表，变难看的是依赖图

把 `call` 换成 attribute，本质是把**显式依赖变成隐式约定**：

| | 显式 call / 构造注入 | attribute 自动扫描 |
|---|---|---|
| Find Usages | 精确 | 失效 |
| 谁依赖谁 | IDE 可查 | 只能靠运行时 log |
| 漏注册 | 编译错 | 静默 / NullRef |
| 新人上手 | 顺着调用链读 | 得先知道“有个扫描器存在” |

大史山从来不是“调用写得丑”造成的，是**模块互相引用 + 没有方向约束 + 抽象过早**造成的。换注册方式只是把山从“显式栈”搬到“隐式注册表”，山还在，还多了一层看不见的土。

### 4. EF Core 这个类比不成立

EF Core 跑在 JIT 服务器环境，AOT 不是它的约束；而且**它自己正在从反射撤退**：Compiled Models、source generators、NativeAOT 支持全是往编译期挪。你拿它当“MS 也这么玩”的证据，其实拿到的是反例。

真正值得抄的是**它的演进方向**，不是它的历史包袱。

### 5. “维护交给 AI” 这条要打折

AI 擅长局部补全和模式复用，不擅长在隐式契约网络里维持全局一致性。而 attribute 魔法恰好是典型的“局部看不出问题、全局才炸”类型——**这正好是 AI 最弱的区间**。它会更积极地帮你加 attribute，然后在你没查的那条分支上漏注册一个。

---

## 我会怎么落地

<details>
<summary><b>展开：分层方案</b></summary>

| 层 | 做法 |
|---|---|
| 装配入口 | 一个手写 `GameBootstrap`，50 行显式注册。比扫 300 个 attribute 好维护，且断点能打断 |
| 生命周期 | 接口 + 列表注册：`ITickable` / `IInitializable`，注册时 `_tickables.Add(x)`。查得见、断得住 |
| 资源 / 网络 / 存档 | 纯接口 + 构造注入，不碰 attribute。Addressables 已经给了资源管理，别自己再造一层反射封装 |
| **引擎回调桥接** | **这里用 attribute 是划算的**：`[Tick]` `[OnUnityAwake]`，因为引擎本来就没给你接口。但必须配 Source Generator，把扫描结果生成成静态注册代码 |
| 字符串反射 | 只允许出现在边界（AssetKey、Bundle 名、序列化字段名），禁止进核心逻辑 |
| 纪律 | asmdef 切模块 + Roslyn analyzer 检查依赖方向（`Core` 不许引用 `UI`）。**这才是治史山的药** |

关键那一条：**attribute 可以留，但让它变成编译期产物**。Source Generator 扫 attribute → 生成 `RegisterAll()`，你得到的是：AOT 安全、可断点、可 Find Usages、IDE 能跳转。运行时反射的收益你拿了，代价几乎全消了。

</details>

---

## 唯一“反射近乎必需”的地方

就是 `MonoBehaviour` 那一侧：引擎创建对象、序列化字段、回调生命周期，这几件事你没有接缝可以插。所以：

- **要给 MonoBehaviour 注入** → 只能反射 / 字段注入
- **要给引擎行为加 hook** → 只能反射扫 attribute 或 Harmony patch

但这两个都是**边界层**的事情，一个项目里应该只有一两百行。**Core 层（资源管理、网络服务、初始化流程）完全不需要反射**，它们的构造权在你手里。

---

## 关于字符串反射

同意你的判断，但补一句：字符串反射的痛是**编译期能感知**的（跑一次就崩、改个名立刻炸）；attribute 自动注册的痛是**编译期无法感知**的（少扫一个类型，五年后才 NullRef）。

前者是“地狱但可见”，后者是“天堂但地雷”。真要选，我选可见的地狱。

---

**一句话总结**：框架变史山跟你用不用反射关系不大，跟你**有没有强制依赖方向**关系极大。反射能帮你少写 wiring，但不会帮你少欠债——它只是把债记在了另一本更难查的账上。

---
## 🧑‍💻 User

怎么说呢，我想把框架设计成类似springboot那套一切都靠注册attribute、event等去交给框架自己管理和调度外加一套config也是框架提供接口，这样上层业务人员就可以专心于业务逻辑的开发

---
## 😀 默认助手 (7)

## 你要的这套东西，Spring Boot 其实不是靠反射撑起来的

先拆解一下 Spring Boot 真正起作用的是哪几层：

| 层 | 是什么 | 能不能搬到 Unity |
|---|---|---|
| 反射/注解扫描 | `@Component` 自动发现 | 能，但这是**最不值钱**的一层 |
| 强类型契约 | 接口 + 构造注入 + 启动期校验 | 能，且必须 |
| Fail-fast 启动 | 缺 bean / 循环依赖 → 启动直接失败并打印依赖图 | **能，而且能做得更好——搬到编译期** |
| 生命周期宿主 | ApplicationContext，单进程、明确 start/stop | **搬不了，这是 Unity 的根本差异** |
| Auto-configuration / starter | 依赖存在就自动装配，条件注册 | 能，但应该最后做，也可能永远不做 |

你项目里“业务人员专心写业务”的收益，80% 来自第二、三行，只有 20% 来自第一行。而大家做框架时，往往把 90% 的精力花在第一行。

---

## 根本差异：Unity 没有 Host

Spring Boot 有一个隐式前提：**一个进程 = 一个应用 = 一次启动 = 一个 ApplicationContext**。

Unity 里没有这个东西。你有：

- 场景加载 / 卸载 / 叠加，对象生命周期是**引擎**给的
- Prefab 实例化，构造权不在你的容器里
- Editor 模式的 domain reload、Enter Play Mode Options（不开就是全量重载，开了就得自己收拾静态状态）
- Addressables 的异步依赖，资源在“启动”时根本没加载
- 多套入口：Boot 场景、测试场景、Editor 直接 Play、CI 里的无头测试

所以你想做的“Spring Boot 那套”，本质是**在 Unity 里造一个 Host，并且要容忍它被反复撕裂重建**。这可以做（VContainer 的 `LifetimeScope` + 显式 `Order` 就是在做这个），但复杂度不会消失，只会转移。转移到的位置叫：**生命周期边界该由谁来划分**。这个答案 Unity 不会给你，得你自己在框架里定义清楚，否则业务人员会写出“我在 Awake 里拿到了一个还没初始化的服务”。

---

## 如果真要做，你要同时付三笔账

### 账 1：可诊断性（最容易被低估的一笔）

Spring Boot 坏了你能看什么？

- 启动失败时打印完整的 bean 依赖链和循环引用路径
- `ConditionEvaluationReport` 告诉你为什么某个 auto-config 没生效
- Actuator 里能看到所有 bean 和它们的来源

**你要是做了 attribute 自动注册，就必须同时做出这三样东西的 Unity 版。** 否则你会得到一个“看不见的”框架：业务人员写了个 service，没生效，然后花两小时不知道为什么。这比他自己写 wiring 慢十倍。

具体要有的东西：

```
[GameBoot] 启动日志（必须默认开，不能是 debug flag）
  ├─ 扫描到 N 个 [GameService]，其中 3 个被条件排除（打印原因）
  ├─ 注册顺序图（Order + 依赖推导）
  ├─ 循环依赖检测 → 抛异常并打印闭环路径
  └─ 未使用的注册（注册了但没人注入）→ 警告
```

最后一条特别值钱：**自动扫描的框架不告诉你“多余”**，而多余注册就是下一个史山的种子。

### 账 2：Domain Reload

Editor 下每次 Play 都会重载 domain（或者你开了 no-reload 就得自己清静态字段）。attribute 注册表如果要缓存到静态变量，就必须做到**幂等重建**。这块 Spring Boot 完全没有对应概念，你抄不到任何东西。

### 账 3：AOT + 裁剪

这个上一条已经说过，这里只强调一件事：**Source Generator 不是可选项，是前提条件。** 运行时 `Assembly.GetTypes()` + `Activator.CreateInstance` 在 IL2CPP + High stripping 下是必然出事的，只是时间问题。

---

## 我建议的形态：契约和 attribute 双轨，运行时零反射

<details>
<summary><b>展开：具体设计（含代码）</b></summary>

### 核心原则

> **attribute 只负责“声明加入自动管辖”，真正的契约由接口保证。**

这样业务人员确实“只写自己的类”，但签名变化仍然是编译期错误。

### 业务人员写的东西

```csharp
// 契约在 Core 里，编译期可查
public interface IBattleService { void StartBattle(in BattleRequest req); }

// 业务人员写的：只有这一个文件
[GameService(Scope = Lifetime.Session, Order = 100)]
public sealed class BattleService : IBattleService, IInitializable, ITickable
{
    private readonly IResourceService _res;
    private readonly GameConfig.Battle _cfg;

    public BattleService(IResourceService res, GameConfig.Battle cfg)
        => (_res, _cfg) = (res, cfg);

    public void Initialize() { /* 此时依赖已就绪 */ }
    public void Tick(float dt) { }
    public void StartBattle(in BattleRequest req) { }
}
```

注意：**没有 `Awake`、没有 `FindObjectOfType`、没有 `GetComponent`、没有单例访问**。这才是“专心业务”。

### Source Generator 生成的注册表

```csharp
// obj/Generated/GameServiceRegistry.g.cs  人可读、可断点、可 Find Usages
public static class GameServiceRegistry
{
    public static void Register(GameContainer c)
    {
        c.AddSingleton<IResourceService, ResourceService>();
        c.AddSingleton<IBattleService, BattleService>(Lifetime.Session, order: 100);
        // 生成器顺手生成 Tick 列表、Initialize 顺序
    }
}
```

### Generator 同时负责 fail-fast（这是关键）

```csharp
// 生成器在编译期发现的问题，直接报 CS 错误，不用等运行
// 1. [GameService] 标在没实现任何框架接口的类上 → 报错
// 2. 构造参数类型没有对应注册 → 报错（这就是 Unity 版的 "no qualifying bean"）
// 3. 循环依赖 → 报错并打印闭环
// 4. Lifetime 违规：Singleton 注入 Session → 报错
// 5. Scope 内注册了但全项目无注入点 → Warning
```

**这就是我最想强调的一点：Spring Boot 的 fail-fast 在启动期，你在 Unity 可以做到编译期。** 比它更好。

### 配置

```csharp
[GameConfig("battle")]                       // 绑定到 Assets/Config/battle.yaml
public sealed partial class BattleConfig     // partial，生成绑定代码
{
    [Range(1, 100)] public int MaxTeamSize { get; init; } = 4;
    public float TimeScale { get; init; } = 1f;
}
```

- 强类型 POCO，不要 `GetConfig("battle.maxTeamSize")` 这种字符串 API
- 加载时**校验失败必须 fail-fast**，且报清楚是哪个文件哪一行
- Editor 里能直接编辑（ScriptableObject 也行，但只要 schema 一致）

### 事件

不要做 `string topic` 的全局总线，那是 Unity 项目第一大隱形耦合源。

```csharp
public readonly struct BattleEnded : IGameEvent { public readonly bool Win; }
public sealed class Xxx : IEventHandler<BattleEnded> { public void OnEvent(in BattleEnded e) {} }
```

`IEventHandler<T>` 由 generator 收集注册。这样“谁在监听”可以用 Find Usages 查，而且事件是**强类型 struct，零装箱**。

</details>

---

## 落地顺序（建议别一次造完）

| 阶段 | 做什么 | 此时业务人员拿到的 |
|---|---|---|
| 1 | 显式 composition root + `IInitializable` / `ITickable` 接口 + 手写注册 | 依赖注入，无反射，可断点 |
| 2 | 加 Source Generator，让 `[GameService]` 生成注册代码 + 编译期校验 | “只写自己的类” |
| 3 | 强类型 config 绑定 + 启动日志 + 依赖图 dump | 能自己排错 |
| 4 | 才考虑条件注册 / auto-configuration / 模块包 | 生态 |

**阶段 2 完成时，你的目标其实已经达成了。** 阶段 4 是锦上添花，也是最容易变成史山的地方——Spring Boot 的 auto-config 之所以不乱，是因为它有 `spring-boot-autoconfigure` 那种级别的考古投入，加上一整个生态在约束。单人/小团队做这个，收益远小于成本。

---

<details>
<summary><b>别做的清单（这些是 Spring Boot 那套在 Unity 里的已知雷区）</b></summary>

1. **别让 attribute 成为唯一契约。** 只标 attribute 不实现接口 → 运行时找不到方法，编译器帮不了你。
2. **别做“自动扫描所有 IEventHandler 并注册”。** 你会在某次裁剪后少几个监听者，然后线上对局数据不对。
3. **别做 `[Inject]` 字段注入作为主路径。** 构造注入能暴露循环依赖，字段注入只会 NullRef。
4. **别让 MonoBehaviour 参与 DI 主流程。** 让它们当 View，逻辑全在纯 C# 类里。这一条能砍掉你一半的反射需求。
5. **别做多套生命周期（Singleton / Scene / Session / Frame）。** 两套够用，超过两套就是给业务人员出选择题。
6. **别在 Core 里出现 `string` 类型的 key。** Addressables 的 key 已经是唯一合法例外。

</details>

---

## 最后一句真心话

你说目标是“业务人员专心业务”。但真正决定他们效率的，往往不是少写几行 wiring，而是**从改代码到看见效果的时间**。

一个 attribute 框架 + 40 秒进 Play，比起手写注册 + 3 秒热重载，业务人员会在后者里写出更好的代码。

所以我的建议是：**框架的“魔法”程度，和你的诊断工具、热重载能力，必须同步增长。** 只加魔法不加透明度，你会在半年后收到一句评价——“这框架比史山还难查”。

---
## 🧑‍💻 User

不知得像et框架、unitygameframework等框架是怎么design，帮我调研一下

---
## 😀 默认助手 (7)

调研完了。先说个反直觉的发现：**这两套框架里没有一个是靠“扫描 assembly 找 attribute”来做业务注册的**——而你这三轮讨论想要的核心机制，恰好是它俩都刻意绕开的。

---

## 一、ET：ECS 化 + 事件驱动 + 分析器强约束

### 核心世界观

ET 的设计文档里有一篇叫《一切皆实体》，讲的是：

> 所有数据都是 Entity，Entity 既可以当成组件使用，也可以当做其它 Entity 的孩子

它拒绝扁平 ECS，改用**树状**结构（理由是“老板不可能直接管几百个工人”）：

```csharp
Head head = human.AddComponent<Head>();
head.AddComponent<Eye>();
head.AddComponent<Ear>();
human.AddComponent<Body>();
```

组件必须走 `ComponentFactory.Create()`，不能 `new`。理由和你的框架设计直接相关——工厂做了三件事：构造、注册到事件系统、抛 `AwakeSystem`、决定是否走对象池。

### 数据 / 逻辑彻底分离

Entity 类**没有方法**，逻辑全在 System 里：

```csharp
public sealed class Player : Entity
{
    public string Account { get; private set; }
    public long UnitId { get; set; }
    // 注意：没有 Awake、没有 Update、没有任何方法
}
```

生命周期和逻辑用 System 类挂上去：

```csharp
[ObjectEvent]
public class PlayerAwakeSystem : AwakeSystem<Player>
{
    public override void Awake(Player self) { }
}
```

### 它为什么明确拒绝反射 —— 这段原文很关键

《组件式设计》里原话：

> **ET 没有像 Unity 使用反射去实现这种功能，因为反射性能比较差，而且这样实现的好处是这个类可以放到热更 dll 中**，这样组件的 Awake、Start、Update 方法以及其它方法都可以放到热更层中。

看清楚这个理由：**性能只是顺带，真正原因是热更**。

Unity 靠反射找 `Awake()` 方法 → 方法必须在主 dll 里 → 改逻辑就得重新打包。ET 用显式的 System 类 → System 类可以塞进热更 dll → 运行时换 dll。

所以 ET 的反射不是被“设计理念”拒绝的，是被**热更这个产品需求**逼走的。你如果不做热更，这条约束对你不成立。

### 事件机制：四层 System + 自定义事件

| 事件 | 触发时机 |
|---|---|
| `AwakeSystem` | 组件创建后，一次 |
| `StartSystem` | Update 前 |
| `UpdateSystem` | 每帧 |
| `DestroySystem` | Dispose 时 |
| `ChangeSystem` | 手动触发 |
| `DeserializeSystem` | 反序列化后（存档读回） |
| `LoadSystem` | 热更 dll 重载时，用于重新注册 handler |

自定义事件就是你说的 attribute：

```csharp
Game.EventSystem.Run("HpChange", oldhp, newhp);

[Event("HpChange")]
public class HpChange_ShowUI : AEvent<int, int>
{
    public override void Run(int a, int b) { }
}
```

**注意这里就是字符串事件名**。ET 自己也在往强类型挪，但历史包袱还在。这是你明确说了要避开的坑，ET 踩了。

### 演化路径：这条线最有参考价值

ET 的注册方式变了三代，方向**一路从运行时反射走向编译期生成**：

| 版本 | 注册方式 |
|---|---|
| ET6 及以前 | 运行时扫 dll 找 `[ObjectSystem]`，建 Type→System 字典 |
| ET8 | 引入 source generator，`EntitySystemOf` 分析器**一键从 Entity 接口生成 System 方法** |
| ET10 | 一整套 analyzer 生态：`et-code`（架构约束）、`et-async`（await 安全）、包依赖分析器、entity 生命周期保护、actor 用法校验 |

ET10 的 README 直接写：analyzer 的作用是"**严格限制 AI 行为**"、"让 AI 在架构边界内工作，而不是生成不受控的代码"。

这基本就是你上一条说的"维护交给 AI"的**正式工程答案**：不是靠 AI 自觉，是靠 analyzer 让 AI 写错就编译不过。

### ET 的代价

- **侵入侵到骨子里**：你不接受 Entity/Component/System 这个世界观，就用不了
- 学习曲线陡到劝退
- 有 LICENSE 限制：开源版**只能学习，不能二次开源，商用 4999**

<details>
<summary><b>ET 值得抄 / 不值得抄</b></summary>

**值得抄：**
- Entity 树 + 组件热插拔（骑马才挂马数据）
- 数据/逻辑分离，逻辑用静态扩展方法（避免"类的虚函数上万行"）
- `InstanceId` 解决对象池 + 异步的经典坑（await 回来后对象已被回收复用）
- 用 source generator 替代手写注册
- 用 analyzer 当架构守卫

**不值得抄：**
- 整套运行时（Fiber/Actor/多进程，除非做 MMO）
- 字符串事件名
- 双端共享代码（除非你有服务端）
</details>

---

## 二、GameFramework / UnityGameFramework：显式分层 + 优先级模块

### 三层结构

```
GameFramework/                 ← 纯 C#，不引用 UnityEngine（可以跑服务端/单元测试）
UnityGameFramework/Runtime/    ← Unity 适配层：GameEntry、各 XxxComponent
游戏逻辑（Procedure / EntityLogic / UIFormLogic 等）
```

19 个模块，每个都是 **`IXxxManager` 接口 + `XxxManager` 实现 + `GameFrameworkModule` 基类**三件套。

### 关键源码：`GameFrameworkEntry`（这是全场最值钱的一段）

```csharp
public static T GetModule<T>() where T : class
{
    Type interfaceType = typeof(T);
    if (!interfaceType.IsInterface)
        throw new GameFrameworkException("You must get module by interface");

    if (!interfaceType.FullName.StartsWith("GameFramework.", StringComparison.Ordinal))
        throw new GameFrameworkException("You must get a Game Framework module");

    string moduleName = Utility.Text.Format("{0}.{1}",
        interfaceType.Namespace,
        interfaceType.Name.Substring(1));        // IEventManager → EventManager

    Type moduleType = Type.GetType(moduleName);
    if (moduleType == null)
        throw new GameFrameworkException($"Can not find module type '{moduleName}'");

    return GetModule(moduleType) as T;
}
```

它做了什么：

1. **只接受接口**，不接受具体类型
2. **只接受 `GameFramework.*` 命名空间**，越界直接抛
3. `IEventManager` → 去 `I` → `EventManager`，靠**命名约定**
4. `Type.GetType` + `Activator.CreateInstance`，**懒创建**
5. 插进链表时按 `Priority` 排序

```csharp
LinkedListNode<GameFrameworkModule> current = s_GameFrameworkModules.First;
while (current != null)
{
    if (module.Priority > current.Value.Priority) break;
    current = current.Next;
}
```

于是 `Update()` 正序轮询，`Shutdown()` **逆序**（`s_GameFrameworkModules.Last` 往回走）。`EventManager.Priority = 7`。

**这就是全部的反射了。** 一次、启动期、限定命名空间、命名约定兜底、失败就抛。这就是我上一轮说的"契约 + 一次反射 + 编译期约束"的极简现实版。

### 依赖获取方式：静态单例，不是注入

```csharp
GameEntry.Event.Fire(this, ShowEntitySuccessEventArgs.Create(...));
```

`GameEntry` 是个 MonoBehaviour，上面挂一堆 `XxxComponent` 序列化引用。**没有容器、没有注入、没有扫描。**

### 事件：int id + 引用池

```csharp
void Subscribe(int id, EventHandler handler);
void Fire(object sender, GameEventArgs e);      // 下一帧分发，线程安全
void FireNow(object sender, GameEventArgs e);   // 立即分发，非线程安全
```

双语义（延迟 / 立即）是很多人踩过的坑，GF 直接给你两个 API。参数对象走 `ReferencePool` 引用计数，**零 GC**。

### Procedure：启动流程 FSM

这是 GameFramework 最像 Spring Boot 的地方——**它把"应用启动"变成一个显式的状态机**：

```
ProcedureLaunch
  → ProcedureSplash
  → ProcedureInitResources
  → ProcedureCheckVersion      ← 热更检查
  → ProcedureUpdateResource
  → ProcedureLoadAssembly      ← HybridCLR
  → ProcedurePreload
  → ProcedureChangeScene
  → ProcedureMenu
```

每个 Procedure 是实现 `ProcedureBase` 的子类，有 `OnEnter / OnUpdate / OnLeave`。**这就是你的 `ApplicationRunner` + `@Order`，而且比 Spring 更可视化**。

---

## 三、正面对比

| 维度 | ET | GameFramework |
|---|---|---|
| 核心抽象 | Entity 树 + System | Manager 模块 + Component |
| 反射用量 | 早期运行时扫 dll → 现在代码生成 | **几乎为零**，仅 module 解析一次 |
| 依赖获取 | 组件查询 / `Game.Scene` | `GameEntry.XXX` 静态单例 |
| 解耦手段 | System 事件 + attribute 订阅 | 事件（int id）+ Procedure FSM |
| 架构约束 | **分析器编译期强制** | 靠代码规范（人肉） |
| 热更 | 一等公民 | 外部方案（HybridCLR/ILRuntime） |
| 侵入性 | 极高（世界观绑定） | 中（模块 + 生命周期接口） |
| 自动扫描注册 | 有（但为热更，且已转向生成） | **无** |
| 适合 | MMO、双端共享、要热更 | 中大型、Unity 重度依赖、要稳 |

---

## 四、对你的 Spring Boot 式设计的三条硬结论

<details>
<summary><b>展开</b></summary>

**结论 1：ET 验证了"attribute + 事件驱动"可行，但它付的账是十年 + 一整套 analyzer**

ET10 的 analyzer 生态（et-code / et-async / 包依赖 / actor 校验 / entity 生命周期）是它敢让 AI 写代码的真正原因。**框架代码不是壁垒，analyzer 才是。** 你想"维护交给 AI"，得先有这个。

**结论 2：GameFramework 验证了"几乎零反射也能撑起 19 个模块"**

它连扫描都不做，只靠 `I` 前缀命名约定 + `Priority` + 接口。代码量小到你能一晚上读完 `GameFrameworkEntry`。

**结论 3：两套框架都没做"扫全 assembly 找 attribute 注册业务"**

ET 扫 dll 是因为热更必须运行时发现，GameFramework 干脆不扫。它们给业务方的统一答案是：

> **给你一个必须继承的基类 / 必须实现的接口，然后模板化。**

`ProcedureBase`、`UIFormLogic`、`EntityLogic`、`AEvent<T>` —— 全是"填空式"，不是"标注式"。**这就是"业务专心写业务"在 Unity 生态的真实验证路径。**

</details>

---

## 五、我建议你怎么取

```
骨架    ← GameFramework：IXxxManager + Priority + Update 顺序 + Shutdown 逆序
启动    ← GameFramework：Procedure 化启动流程（Spring 的 ApplicationRunner 等价物）
数据    ← ET：Entity 树 + 组件热插拔 + 数据/逻辑分离
注册    ← ET：attribute → Source Generator → 生成静态注册表（不要运行时扫描）
约束    ← ET10：analyzer 当架构守卫，写错编译不过
事件    ← 两者混：GF 的 Fire/FireNow 双语义 + 强类型 GameEventArgs + ReferencePool
           ✗ 不要 ET 的 [Event("HpChange")] 字符串
✗ 不要  ← ET 的 Fiber/Actor/多进程运行时（除非你是 MMO）
```

**一句话**：你要的东西 = **GameFramework 的骨头 + ET 的生成器 + 你自己写的 analyzer**。缺 analyzer 那一块，就是你说的"大史山"的入口——因为没有任何东西会拦住错误写法。

---

## 六、建议你直接去读的部分

| 来源 | 路径 | 为什么 |
|---|---|---|
| GameFramework | `GameFramework/Base/GameFrameworkEntry.cs` | **100 行，最值钱**，模块系统全貌 |
| GameFramework | `GameFramework/Event/EventManager.cs` + `EventPool` | 双语义 + 引用池零 GC |
| GameFramework | `GameFramework/Procedure/` | 启动流程 FSM |
| ET | `Book/4.1组件式设计.md` | 为什么拒绝反射的原文理由 |
| ET | `Book/3.3一切皆实体.md` | 树状 Entity、InstanceId 坑 |
| ET | `Book/3.4事件机制EventSystem.md` | 7 种 System + 自定义事件 |
| ET | `Book/8.1ET Package制作指南.md` | 包化 + 依赖约束（你对标的"starter"） |

ET 的 Book 是中文原版在 `Book/` 下，英文版是 `xxx.md`、中文是 `xxx中文.md` 并存，别下错。

需要的话，我可以把 GameFramework 那套 `IXxxManager + Priority` 骨架和 Generator 版本的 attribute 注册拼成一个最小可跑原型，给你对着改。
