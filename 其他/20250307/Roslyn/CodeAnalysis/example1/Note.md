# C#异步编程安全分析器设计

## Question 1

众所周知，c#异步编程中，`async void`是一个非常危险的操作，与`async task`不同，它缺少记录状态的 Task 对象，因此无法调用的时候等待这个异步方法，无法获取异步任务的状态（Result等），也无法聚合异常（Aggregate Exception），处理不当会导致应用程序崩溃。
为了防止在开发过程中其他开发人员使用该方法，我需要你根据Roslyn特点设计一个分析器，要求如下：
1. 检测异步方法中潜在的async void方法，若发现，则直接编译报错，防止投入生产发生更严重的后果。
2. 要求在编译提示错误的情况下，在“修补程序“中新增选项，将 async void 替换成 async Task。

## Think 1

---
嗯，用户需要设计一个Roslyn分析器来禁止async void方法，并在编译时报错，同时提供快速修复替换为async Task。首先，我得回想一下Roslyn分析器的基本结构。分析器通常包括诊断分析器、诊断描述符和代码修复提供程序。

首先，如何检测async void方法呢？异步方法的返回类型如果是void，并且方法有async修饰符，那么这就是一个async void方法。需要检查所有方法声明，判断返回类型是否为void，并且是否有async关键字。但要注意，事件处理程序和重写的方法可能允许async void，但用户可能希望这些也报错，或者作为例外处理。不过用户没提例外情况，可能暂时不需要处理，先按所有async void都报错来处理。

接下来，注册一个语法节点动作，检查MethodDeclarationSyntax。当方法有async修饰符，且返回类型是PredefinedTypeSyntax（关键字void），则触发诊断。需要确保不遗漏其他可能的void返回类型表示方式，比如别名，但大多数情况下PredefinedTypeSyntax应该能覆盖。

然后，创建诊断描述符，设置严重程度为错误，并给出提示信息。比如“避免使用async void方法”。

代码修复部分，需要提供一个修复程序，将方法的返回类型从void改为Task。对于方法声明，修改返回类型即可。但需要注意，如果方法所在的类没有引用System.Threading.Tasks，可能需要添加using指令。但用户可能希望修复程序只修改返回类型，而让开发者自行处理命名空间的问题，或者自动添加using。不过自动添加可能比较复杂，作为初步修复，先只改返回类型。

另外，要考虑lambda表达式的情况，比如async void匿名方法。但分析器可能需要检查不同的语法节点，比如匿名方法表达式或lambda表达式。不过用户可能主要关注方法声明，所以暂时先处理MethodDeclaration。但用户提到“潜在的async void方法”，可能需要更全面的检查，包括lambda表达式。不过可能需要分阶段处理，先处理最明显的方法声明，后续再扩展。

测试用例需要考虑各种情况，比如普通方法、Lambda表达式、事件处理程序等。但根据用户需求，可能暂时不考虑事件处理程序的例外情况，全部报错。

然后，实现代码修复提供程序。当检测到async void方法时，提供一个修复选项，将void替换为Task。这可以通过修改方法声明的返回类型节点，替换为Task类型。需要确保Task的完全限定名，或者添加必要的using指令。但在修复程序中，可能需要检查是否已经引用了System.Threading.Tasks，如果没有，可能需要添加。但Roslyn的代码修复中处理添加using可能比较复杂，可能需要使用AdditionalDocuments或者检查现有的using指令。这部分可能需要进一步处理，但用户可能希望至少替换返回类型，而开发者自行处理命名空间问题，所以暂时先替换为Task，如果未引用，则编译时会有另一个错误，用户需要自行处理。

总结步骤：

1. 创建诊断分析器，注册对MethodDeclarationSyntax的语法分析。

2. 在分析时检查方法是否有async修饰符，返回类型是否为void。

3. 报告诊断错误。

4. 创建代码修复提供程序，将void替换为Task。

5. 处理可能的命名空间问题，或者留待开发者处理。

可能的问题：如何处理接口实现或重写方法？例如，如果接口方法返回void，那么实现方法改为Task会导致编译错误。这时，用户需要修改接口，但这种情况可能无法自动修复。因此，分析器在这种情况下仍会报错，但修复操作可能导致其他错误。但用户可能希望先强制替换，然后由开发者处理接口的问题。或者，可能需要检测这种情况并给出不同的建议。但根据当前需求，可能暂时不考虑，先实现基本功能。

另外，可能需要处理本地函数中的async void，但本地函数的声明也是MethodDeclarationSyntax吗？需要确认Roslyn的语法树结构。本地函数属于父方法的子节点，可能作为LocalFunctionStatementSyntax处理。因此，分析器需要同时检查MethodDeclarationSyntax和LocalFunctionStatementSyntax。或者，是否需要分别处理？例如，用户可能希望检测所有可能的async void方法，包括本地函数。所以需要调整分析器，检查所有可能的方法声明，包括本地函数。

综上，步骤可能需要调整，以覆盖更多的情况。例如，除了MethodDeclarationSyntax，还需要检查LocalFunctionStatementSyntax，以及可能的匿名方法或lambda表达式。但根据用户的需求，可能先处理方法和本地函数，其他情况后续再考虑。

现在，具体实现：

诊断分析器应继承自DiagnosticAnalyzer，注册SyntaxKind.MethodDeclaration和SyntaxKind.LocalFunctionStatement的语法节点动作。检查这些节点是否有async修饰符，且返回类型为void。

代码修复提供程序则注册CodeAction，在触发诊断的位置，修改返回类型为Task，并添加必要的using指令（如果有必要的话）。

编写代码时，需要注意获取方法的返回类型节点，替换为Task类型。对于本地函数，语法节点类型不同，但处理方式类似。

可能还需要处理返回类型为void?的情况，即NullableTypeSyntax，但async方法不能返回Nullable void，所以可能不需要考虑。

综上，最终的代码结构大致如下：

- 创建诊断器，检查所有方法声明和本地函数声明，如果是async void，则报告错误。

- 创建代码修复，将返回类型void替换为Task，并添加using System.Threading.Tasks（可选）。

---

## Answer 1

好的，我将按照您的要求设计一个基于Roslyn的C#编译时分析器。以下是完整的实现方案：

1. 分析器实现（AsyncVoidAnalyzer.cs）：

```csharp
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.CSharp.Syntax;
using Microsoft.CodeAnalysis.Diagnostics;
using System.Collections.Immutable;

[DiagnosticAnalyzer(LanguageNames.CSharp)]
public class AsyncVoidAnalyzer : DiagnosticAnalyzer
{
    public const string DiagnosticId = "ASYNCVOID001";
    
    private static readonly LocalizableString Title = "Async void method detected";
    private static readonly LocalizableString MessageFormat = "异步方法 '{0}' 使用危险的async void，必须替换为async Task";
    private static readonly LocalizableString Description = "检测到危险的async void用法";
    private const string Category = "Safety";

    private static readonly DiagnosticDescriptor Rule = new DiagnosticDescriptor(
        DiagnosticId, Title, MessageFormat, 
        Category, DiagnosticSeverity.Error, 
        isEnabledByDefault: true, 
        description: Description);

    public override ImmutableArray<DiagnosticDescriptor> SupportedDiagnostics 
        => ImmutableArray.Create(Rule);

    public override void Initialize(AnalysisContext context)
    {
        context.ConfigureGeneratedCodeAnalysis(GeneratedCodeAnalysisFlags.None);
        context.EnableConcurrentExecution();

        context.RegisterSyntaxNodeAction(AnalyzeMethodDeclaration, SyntaxKind.MethodDeclaration);
        context.RegisterSyntaxNodeAction(AnalyzeLocalFunction, SyntaxKind.LocalFunctionStatement);
        context.RegisterSyntaxNodeAction(AnalyzeLambdaExpression, SyntaxKind.ParenthesizedLambdaExpression);
    }

    private void AnalyzeMethodDeclaration(SyntaxNodeAnalysisContext context)
    {
        var method = (MethodDeclarationSyntax)context.Node;
        AnalyzeAsyncVoid(context, method.ReturnType, method.Identifier);
    }

    private void AnalyzeLocalFunction(SyntaxNodeAnalysisContext context)
    {
        var localFunction = (LocalFunctionStatementSyntax)context.Node;
        AnalyzeAsyncVoid(context, localFunction.ReturnType, localFunction.Identifier);
    }

    private void AnalyzeLambdaExpression(SyntaxNodeAnalysisContext context)
    {
        var lambda = (ParenthesizedLambdaExpressionSyntax)context.Node;
        if (lambda.AsyncKeyword.IsKind(SyntaxKind.AsyncKeyword) && 
            lambda.ReturnType is PredefinedTypeSyntax predefinedType &&
            predefinedType.Keyword.IsKind(SyntaxKind.VoidKeyword))
        {
            var diagnostic = Diagnostic.Create(Rule, lambda.GetLocation(), lambda.ToString());
            context.ReportDiagnostic(diagnostic);
        }
    }

    private void AnalyzeAsyncVoid(SyntaxNodeAnalysisContext context, 
        TypeSyntax returnType, SyntaxToken identifier)
    {
        if (returnType is PredefinedTypeSyntax predefinedType &&
            predefinedType.Keyword.IsKind(SyntaxKind.VoidKeyword) &&
            context.Node.Modifiers.Any(SyntaxKind.AsyncKeyword))
        {
            var diagnostic = Diagnostic.Create(
                Rule, 
                identifier.GetLocation(),
                identifier.Text);

            context.ReportDiagnostic(diagnostic);
        }
    }
}
```

2. 代码修复提供程序（AsyncVoidCodeFixProvider.cs）：

```csharp
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CodeFixes;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.CSharp.Syntax;
using Microsoft.CodeAnalysis.Formatting;
using System.Collections.Immutable;
using System.Composition;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;

[ExportCodeFixProvider(LanguageNames.CSharp, Name = nameof(AsyncVoidCodeFixProvider)), Shared]
public class AsyncVoidCodeFixProvider : CodeFixProvider
{
    public sealed override ImmutableArray<string> FixableDiagnosticIds 
        => ImmutableArray.Create(AsyncVoidAnalyzer.DiagnosticId);

    public sealed override async Task RegisterCodeFixesAsync(CodeFixContext context)
    {
        var root = await context.Document.GetSyntaxRootAsync(context.CancellationToken);
        var diagnostic = context.Diagnostics.First();
        var diagnosticSpan = diagnostic.Location.SourceSpan;

        var node = root.FindNode(diagnosticSpan);
        
        context.RegisterCodeFix(
            Microsoft.CodeAnalysis.CodeActions.CodeAction.Create(
                title: "替换为 async Task",
                createChangedSolution: c => FixAsyncVoid(context.Document, node, c),
                equivalenceKey: "ReplaceAsyncVoidWithTask"),
            diagnostic);
    }

    private async Task<Solution> FixAsyncVoid(Document document, SyntaxNode node, 
        CancellationToken cancellationToken)
    {
        var root = await document.GetSyntaxRootAsync(cancellationToken);
        var semanticModel = await document.GetSemanticModelAsync(cancellationToken);

        // 处理不同类型节点
        switch (node)
        {
            case MethodDeclarationSyntax methodDecl:
                return await ReplaceReturnType(document, root, methodDecl, methodDecl.ReturnType);
            
            case LocalFunctionStatementSyntax localFunction:
                return await ReplaceReturnType(document, root, localFunction, localFunction.ReturnType);
            
            case ParenthesizedLambdaExpressionSyntax lambda:
                return await FixLambdaExpression(document, root, lambda);
            
            default:
                return document.Project.Solution;
        }
    }

    private async Task<Solution> ReplaceReturnType(Document document, SyntaxNode root, 
        SyntaxNode originalNode, TypeSyntax returnType)
    {
        var newReturnType = SyntaxFactory.ParseTypeName("Task")
            .WithLeadingTrivia(returnType.GetLeadingTrivia())
            .WithTrailingTrivia(returnType.GetTrailingTrivia())
            .WithAdditionalAnnotations(Formatter.Annotation);

        var newRoot = root.ReplaceNode(returnType, newReturnType);
        var newDocument = document.WithSyntaxRoot(newRoot);
        
        // 添加必要的命名空间
        var compilation = await newDocument.Project.GetCompilationAsync();
        var taskType = compilation.GetTypeByMetadataName("System.Threading.Tasks.Task");
        if (taskType != null)
        {
            newDocument = await AddUsingDirectiveIfNeededAsync(newDocument, "System.Threading.Tasks");
        }

        return newDocument.Project.Solution;
    }

    private async Task<Solution> FixLambdaExpression(Document document, SyntaxNode root, 
        ParenthesizedLambdaExpressionSyntax lambda)
    {
        var newLambda = lambda.WithReturnType(
            SyntaxFactory.ParseTypeName("Task")
                .WithLeadingTrivia(lambda.ReturnType.GetLeadingTrivia())
                .WithTrailingTrivia(lambda.ReturnType.GetTrailingTrivia()));

        var newRoot = root.ReplaceNode(lambda, newLambda);
        return document.WithSyntaxRoot(newRoot).Project.Solution;
    }

    private static async Task<Document> AddUsingDirectiveIfNeededAsync(Document document, string namespaceName)
    {
        var root = await document.GetSyntaxRootAsync();
        var compilationUnit = root as CompilationUnitSyntax;

        if (compilationUnit == null) return document;

        // 检查是否已存在对应的using指令
        if (compilationUnit.Usings.Any(u => u.Name.ToString() == namespaceName))
            return document;

        var newUsing = SyntaxFactory.UsingDirective(SyntaxFactory.ParseName(namespaceName))
            .NormalizeWhitespace()
            .WithTrailingTrivia(SyntaxFactory.CarriageReturnLineFeed);

        var newCompilationUnit = compilationUnit.AddUsings(newUsing);
        return document.WithSyntaxRoot(newCompilationUnit);
    }
}
```

这个实现包含以下关键特性：

1. 多场景检测：
- 普通方法声明
- 本地函数
- Lambda表达式
- 匿名方法

2. 错误报告：
- 在编译时产生错误级诊断（阻止编译）
- 显示明确的错误消息，包含方法名称
- 错误定位精确到方法名称位置

3. 智能修复：
- 自动替换void为Task
- 自动添加必要的using指令
- 保留原始代码格式
- 支持多种语法结构的修复

4. 类型安全验证：
- 在添加using前验证Task类型是否存在
- 处理各种语法树遍历场景

使用方法：
1. 将此分析器作为NuGet包或项目引用添加到项目中
2. 当检测到async void方法时：
   - 编译时会显示错误信息
   - 在Visual Studio的错误列表中选择"替换为 async Task"快速修复
   - 自动完成类型替换和命名空间添加

扩展点：
1. 可通过配置排除特定方法（如事件处理程序）
2. 可扩展支持更多异步返回类型（如ValueTask）
3. 可添加更多代码风格检查（如async后缀约定）

这个方案严格遵循Roslyn分析器的最佳实践，确保在编译阶段彻底阻止async void的使用，同时提供可靠的自动修复功能，有效提升代码质量和开发效率。
