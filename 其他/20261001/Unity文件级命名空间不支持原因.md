# Unity文件级命名空间不支持原因

## 🧑‍💻 User

unity为什么不能写namespace xxx;这种

---
## 😀 默认助手 (7)

## 结论

`namespace Foo;` 这种写法叫 **file-scoped namespace（文件级命名空间）**，是 **C# 10** 才引入的语法。而 Unity 使用的 C# 语言版本被**固定为 C# 9.0**。即使是最新的 Unity 6.x，官方文档仍写着 C# 9.0。因此，编译器一看到文件作用域的 namespace，就会直接报错：

```
error CS8773: Feature 'file-scoped namespace' is not available in C# 9.0.
Please use language version 10.0 or greater.
```

这是**语言版本限制**造成的；Unity 并不禁止使用 namespace。C# 9 里写 `namespace Foo { ... }` 完全没问题。

## 无法通过添加编译参数解决的原因

Unity 并不使用你系统安装的 .NET SDK 来编译脚本，它使用的是**自己随 Editor 打包的 Roslyn 版本**（对应 C# 9 时代，约 Roslyn 3.8/3.9 一代）。所以：

- 在 `csc.rsp` 里写 `-langversion:10` **没用**：甚至可能直接报 `CS1617: Invalid option '10' for /langversion`，因为那个编译器构建版本根本没有 `10` 这个选项；
- 写 `-langversion:preview` 也没用，因为 C# 10 的功能在那一代 Roslyn 里**根本没有实现**，作用域不包含它，因此编译不通过；
- 装个 nuget 包、改 asmdef 里 `LangVersion` 也不行，Unity 的脚本编译管线不允许覆盖。

也就是说，这是编译器能力的问题，靠配置无法解决。

## 实际可行的做法

```csharp
namespace MyGame.Combat
{
    public class DamageDealer : MonoBehaviour
    {
        void Hit() { }
    }
}
```

能做的只是减少无谓重复：

1. **用 IDE 的文件模板或代码片段（snippet）**：Rider、VS 新建文件时自动生成带块级 namespace 的骨架，实际输入量差别不大。
2. **层级深的项目用 `namespace A.B.C { }` 一次写完**，不必逐层嵌套。
3. **使用 Unity 的 `Assembly Definition` 划分模块**，从组织角度减少对命名空间的依赖，这是 Unity 生态里更“原生”的做法。

<details>
<summary>顺便列出：Unity 里同样用不了的 C# 10+ 特性（来自同一批编译器限制）</summary>

| 特性 | 引入版本 |
|---|---|
| `namespace X;` 文件级命名空间 | C# 10 |
| `global using` 和隐式 using | C# 10 |
| `record struct`、无参构造函数的结构体 | C# 10 |
| 常量内插字符串 | C# 10 |
| 文件内 `using` 别名增强 | C# 12 |
| `required` 成员 | C# 11 |
| 原始字符串 `"""..."""` | C# 11 |
| 类的主构造函数、集合表达式 `[1,2,3]` | C# 12 |

另外，官方文档明确列出 C# 9 中**连 C# 9 本身的一部分功能也没实现**：
- `init` only setters（`init` 访问器）
- Covariant return types（协变返回类型）
- Module initializers（`[ModuleInitializer]`）
- 抑制 `localsinit` 标志
- unmanaged 函数指针的可扩展调用约定

`record` 只能“半用”：需要自己手写 `System.Runtime.CompilerServices.IsExternalInit` 类型，而且 Unity 序列化系统不认 record，所以不要用 record 做序列化数据。

</details>

## 关于未来

Unity 一直在推进把运行时换成 CoreCLR / .NET 8+（Unity 6.x 之后的路线图里反复提到），真落地后才可能解锁 C# 10+。但在官方明确宣布支持并写进文档之前，**请按 C# 9 来写代码**，否则你在编辑器里编译通过、一打包到 IL2CPP 或别的平台就出问题，交叉编译的一致性才是 Unity 锁版本的根本原因。
