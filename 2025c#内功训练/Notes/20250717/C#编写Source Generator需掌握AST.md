# C#编写Source Generator需掌握AST

### 🧑‍💻 User

在c#中，编写sourcegenerator需要了解ast吗？

---

### 🤖 Assistant

在C#中编写Source Generator时，**理解抽象语法树（AST）是必需的**，但根据场景不同，深入程度会有所差异。以下是关键要点：

### 1. **基础场景（可忽略AST）**
   - 若仅需**生成全新代码**（如基于配置文件的DTO类），可直接拼接字符串或使用`StringBuilder`等模板技术，无需解析现有代码。
   - 示例：从JSON文件生成静态类。

### 2. **进阶场景（必须掌握AST）**
   当需要**分析现有代码结构**时，必须使用Roslyn的AST API：
   - **识别特定语法元素**  
     如查找标记了`[GenerateInterface]`的类，需遍历AST检查属性节点。
   - **提取代码语义**  
     如读取类的属性/方法信息生成扩展代码。
   - **避免重复生成**  
     检查目标类是否已存在特定成员。

### 3. **关键AST操作示例**
   ```csharp
   // 在Source Generator中获取所有类语法节点
   public void Execute(GeneratorExecutionContext context)
   {
       var syntaxTrees = context.Compilation.SyntaxTrees;
       foreach (var tree in syntaxTrees)
       {
           var root = tree.GetRoot();
           var classes = root.DescendantNodes().OfType<ClassDeclarationSyntax>();
           foreach (var cls in classes)
           {
               // 检查类上的特性
               var attributes = cls.AttributeLists.SelectMany(a => a.Attributes);
               if (attributes.Any(a => a.Name.ToString() == "GenerateInterface"))
               {
                   // 基于AST信息生成代码
                   string generatedCode = GenerateInterfaceForClass(cls);
                   context.AddSource($"{cls.Identifier}.g.cs", generatedCode);
               }
           }
       }
   }
   ```

### 4. **必须掌握的核心概念**
   - **语法节点（SyntaxNode）**  
     如`ClassDeclarationSyntax`、`MethodDeclarationSyntax`。
   - **语义模型（SemanticModel）**  
     通过`compilation.GetSemanticModel(tree)`获取类型符号等深层信息。
   - **SyntaxWalker/SyntaxVisitor**  
     高效遍历AST的工具类。

### 5. **推荐学习路径**
   1. 使用Visual Studio的**Syntax Visualizer**（视图 → 其他窗口 → Syntax Visualizer）实时查看AST结构。
   2. 学习Roslyn API：
      - `SyntaxNode` 基础类型体系
      - `CSharpSyntaxWalker` 遍历模式
      - `ISymbol` 语义分析
   3. 参考官方示例：
      [GitHub: dotnet/roslyn-sdk](https://github.com/dotnet/roslyn-sdk/tree/main/samples/CSharp/SourceGenerators)

### 结论
- 对于**仅生成新代码**的简单生成器，可不接触AST。
- 对于**需要分析或修改现有代码**的生成器（90%场景），**AST是核心技能**。Roslyn的AST API是Source Generator与编译过程交互的基础，深入理解能显著提升生成器能力。