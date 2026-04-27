# Java与C#面向对象对比

### 🧑‍💻 User

在java面向对象编程中，有哪些基础题常见考点，对于熟悉c#的同学，哪些是可以融会贯通的，哪些是有明显差异的（例如泛型）？列举常见30个

---

### 🤖 Assistant

对于熟悉 C# 的开发者来说，学习 Java 的面向对象编程（OOP）会非常轻松，因为两者在最初的设计哲学上高度相似。然而，随着语言的发展，两者在底层机制和高级特性上产生了明显的分歧。

为了高效对比，我将这 30 个常见考点分为三大类：**可以直接融会贯通的（概念通用）**、**有语法/细节差异的**，以及**有明显核心差异的（重点关注）**。

为了提升阅读体验，对于您可能已经非常熟悉的通用概念，我使用了折叠面板，您可以重点阅读展开的差异部分。

---

<details open>
<summary><h3>一、 核心机制差异：C# 开发者必须警惕的“坑”（第 1-10 题）</h3></summary>
这些是面试中最常考，也是 C# 开发者最容易按照惯性思维写错的地方。

**1. 泛型机制（Generics）：类型擦除 vs 具现化**
*   **Java**: 采用**类型擦除 (Type Erasure)**。在编译后，`List<String>` 和 `List<Integer>` 在运行时都会变成 `List`（也就是 `Object`）。因此，无法在运行时通过反射获取泛型的实际类型，也不能直接 `new T()`。
*   **C#**: 采用**具现化泛型 (Reified Generics)**。泛型类型在运行时是保留的，性能更好，且支持值类型泛型。

**2. 枚举（Enum）：全功能类 vs 值类型**
*   **Java**: `enum` 是一个完整的类。它可以有成员变量、构造函数、方法，甚至实现接口。
*   **C#**: `enum` 是基于整数的值类型（底层通常是 `int`），功能相对简单。

**3. 值类型与引用类型（Structs vs Primitives）**
*   **Java**: 只有 8 种基本数据类型（`int`, `double` 等）是分配在栈上的。**没有自定义的值类型（Struct）**。所有的对象都是引用类型（分配在堆上）。
*   **C#**: 有统一的类型系统，开发者可以使用 `struct` 自定义值类型。

**4. 异常处理：Checked vs Unchecked Exceptions**
*   **Java**: 独有 **Checked Exception（受检异常）**。例如 `IOException`，编译器强制要求开发者必须使用 `try-catch` 捕获，或者在方法签名上使用 `throws` 声明抛出。
*   **C#**: 只有 Unchecked Exception，编译器从不强制要求捕获异常。

**5. 属性（Properties）：Getters/Setters vs 原生属性**
*   **Java**: 没有原生属性语法。必须手动或通过工具（如 Lombok）编写 `getX()` 和 `setX()` 方法。
*   **C#**: 有优雅的原生属性 `{ get; set; }`。

**6. 方法的虚实（Virtual Methods）：默认虚方法 vs 显式声明**
*   **Java**: 所有非 `private`、非 `static` 的方法**默认都是虚方法**，可以直接被子类重写。如果不想被重写，必须加 `final` 关键字。
*   **C#**: 方法默认是非虚的。必须显式添加 `virtual`，子类显式添加 `override` 才能重写。

**7. 委托与事件（Delegates & Events）**
*   **Java**: **没有委托和事件的概念**。Java 通过单一抽象方法接口（Functional Interfaces，如 `Runnable`, `Consumer`）和匿名内部类/Lambda 表达式来实现类似回调的功能。
*   **C#**: 原生支持 `delegate` 和 `event`。

**8. 字符串相等性比较（String Equality）**
*   **Java**: `==` 比较的是**引用地址**。比较字符串内容必须使用 `str1.equals(str2)`。
*   **C#**: `==` 操作符被重载了，直接用于比较字符串的**值**。

**9. 参数传递（Pass by Reference/Value）**
*   **Java**: **只有按值传递（Pass by Value）**。对于对象，传递的是对象引用的副本。Java 中没有 C# 的 `ref` 或 `out` 关键字。
*   **C#**: 支持按值传递，以及通过 `ref` 和 `out` 实现按引用传递。

**10. 扩展方法（Extension Methods）**
*   **Java**: 没有扩展方法。通常通过编写包含静态方法的工具类（如 `StringUtils`）来实现。
*   **C#**: 可以在静态类中使用 `this` 关键字为现有类型添加扩展方法。

</details>

---

<details open>
<summary><h3>二、 语法与细节差异：稍微调整习惯即可（第 11-20 题）</h3></summary>
这些概念在两门语言中都有，但关键字或用法略有不同。

**11. 万能修饰符：`final` vs `sealed`/`readonly`/`const`**
*   **Java**: `final` 是个多面手。修饰类等于 C# 的 `sealed`（不可继承）；修饰方法代表不可被 `override`；修饰变量等于 C# 的 `readonly` 或 `const`。
*   **C#**: 针对不同场景有细分的关键字（`sealed`, `readonly`, `const`）。

**12. 继承与实现接口的语法**
*   **Java**: 区分关键字。继承类用 `extends`，实现接口用 `implements`。
    ```java
    class Dog extends Animal implements IPet { }
    ```
*   **C#**: 统一使用冒号 `:`。

**13. 调用父类（Base Class Invocation）**
*   **Java**: 使用 `super` 调用父类方法，使用 `super()` 调用父类构造函数。
*   **C#**: 使用 `base` 和 `: base()`。

**14. 访问修饰符的默认可见性**
*   **Java**: 如果不写修饰符，默认是 **Package-Private（包内可见）**。
*   **C#**: 如果不写，类默认是 `internal`（程序集可见），类成员默认是 `private`。

**15. 内部类（Inner Classes）的引用机制**
*   **Java**: 非静态内部类会**隐式持有外部类的引用**。如果不需要访问外部类实例，必须声明为 `static class`（静态嵌套类）。
*   **C#**: 嵌套类默认就是像 Java 的静态嵌套类一样，不持有外部类的引用。

**16. 资源清理：`try-with-resources` vs `using`**
*   **Java**: 实现了 `AutoCloseable` 接口的类，可以使用 `try (Resource r = new Resource()) { ... }` 自动清理。
*   **C#**: 实现了 `IDisposable` 的类，使用 `using` 语句。

**17. 数组协变（Array Covariance）与泛型集合**
*   在 Java 和 C# 中，对象数组都是协变的。但 Java 的泛型集合不支持原生协变，必须使用通配符 `? extends T`；C# 支持接口级别的协变和逆变（如 `IEnumerable<out T>`）。
*   *注：Java 中 HashMap 的查找时间复杂度平均为 $O(1)$，最坏情况退化为红黑树时为 $O(\log n)$。*

**18. 类型转换与判断（Type Casting & Checking）**
*   **Java**: `if (obj instanceof String)`。Java 16 引入了模式匹配：`if (obj instanceof String s)`。
*   **C#**: `if (obj is String s)` 或使用 `as` 关键字进行安全转换。

**19. 初始化块（Initialization Blocks）**
*   **Java**: 支持实例初始化块 `{ ... }` 和静态初始化块 `static { ... }`，在构造函数之前运行。
*   **C#**: 通常只使用构造函数和静态构造函数。

**20. 注解 vs 特性（Annotations vs Attributes）**
*   **Java**: 称为 Annotation，使用 `@` 符号，如 `@Override`。
*   **C#**: 称为 Attribute，使用方括号，如 `[Obsolete]`。

</details>

---

<details>
<summary><h3>三、 可以完全融会贯通的基础考点（点击展开查看 第 21-30 题）</h3></summary>

对于 C# 开发者，以下这 10 个 Java 面试常见考点，您只需要复习一下概念即可，因为它们的底层逻辑和代码组织方式在两种语言中几乎一模一样。

**21. 面向对象三大特性：** 封装（Encapsulation）、继承（Inheritance）、多态（Polymorphism）的定义和作用。
**22. 类与对象：** `class` 的定义，使用 `new` 关键字实例化对象。
**23. 构造函数（Constructors）：** 构造函数的重载，默认无参构造函数的作用。
**24. 方法重载（Method Overloading）：** 同名方法通过不同的参数签名（参数个数、类型）进行区分。
**25. 静态成员（Static Members）：** `static` 变量和方法属于类级别，而非实例级别。
**26. 接口（Interfaces）：** 定义方法契约。Java 8 和 C# 8 都引入了接口中的默认方法（Default Methods）。
**27. 抽象类（Abstract Classes）：** 不能被实例化，使用 `abstract` 修饰包含未实现方法的类。
**28. 装箱与拆箱（Boxing and Unboxing）：** 基础类型（如 Java 的 `int`，C# 的 `int`）与其包装类（如 Java 的 `Integer`，C# 的 `Object` 或 `ValueType`）之间的自动转换。
**29. 垃圾回收（Garbage Collection, GC）：** 开发者不需要手动释放内存，由后台 GC 线程负责回收不可达对象（标记-清除算法等概念通用）。
**30. `this` 关键字：** 用于指代当前对象的实例，常用于解决成员变量与局部变量同名的问题。

</details>

### 给 C# 转 Java 者的备考建议
如果您要去面试 Java 岗位，**请把精力主要放在第一类（前 10 题）上**。面试官非常喜欢问类似 *"Java 的泛型擦除带来了什么影响？"* 或者 *"String 的 `==` 和 `equals` 有什么区别？"*。在写手撕代码时，切记不要给字符串用 `==`，以及习惯手动写 Get/Set 方法（或者熟练使用 Lombok 注解）。