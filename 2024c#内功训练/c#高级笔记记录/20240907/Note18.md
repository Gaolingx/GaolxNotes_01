# C#高级编程之——单元测试（一）基本介绍与使用

## 一、为什么需要单元测试

在我们之前，测试某些功能是否能够正常运行时，我们都将代码写到Main方法中，当我们测试第二个功能时，我们只能选择将之前的代码清掉，重新编写。此时，如果你还想重新测试你之前的功能时，这时你就显得有些难为情了，因为代码都被你清掉了。当然你完全可以把代码写到一个记事本中进行记录，但是这样总归没有那么方便。当然你也可以重新新建一个项目来测试新的功能，但随着功能越来越多，重新新建项目使得项目越来越多，变得不易维护，此时你若选择使用单元测试功能，就可以完美解决你的困扰。

NUnit 提供了单元测试能力，也是目前用的比较流行的单元测试组件。

## 二、什么是NUnit？

NUnit是一个专为.NET平台设计的开源单元测试框架，它属于xUnit家族的一员，最初从 JUnit 移植，当前的生产版本 3 已完全重写，具有许多新功能和对各种 .NET 平台的支持。
作为适用于所有 .Net 语言的单元测试框架，与JUnit（Java）、CPPUnit（C++）等测试框架有着相似的架构和理念。NUnit通过提供一系列丰富的工具和特性，帮助开发者编写、组织和运行单元测试，以确保.NET应用程序的各个部分按预期工作。

## 三、NUnit 使用

**3.1 步骤：**
打开Nuget包管理器，通过Nuget 安装 如下三个包：

- NUnit
- NUnit3TestAdapter
- Microsoft.NET.Test.Sdk

注：

- NUnit 包应由每个测试程序集引用，但不应由任何其他测试程序集引用。
- NUnit使用自定义特性来标识测试。 所有的NUnit属性都包含在NUnit中的 框架的命名空间。 包含测试的每个源文件必须包含该名称空间的using语句（NUnit.Framework），项目必须引用框架程序集nunit.framework.dll。

## 四、创建你的第一个单元测试

为了标识出你的测试用例，你需要在需要进行单元测试的方法前加上 Test Attribute，标记表示测试的 TestFixture 的方法。

其余相关的Attribute参考如下图：

<div class="table-wrapper"><table class="md-table">
<thead>
<tr class="md-end-block"><th><span class="td-span"><span class="md-plain">Attribute</span></span></th><th><span class="td-span"><span class="md-plain">Usage</span></span></th></tr>
</thead>
<tbody>
<tr class="md-end-block">
<td><span class="td-span"><span class="md-meta-i-c  md-link"><a href="https://docs.nunit.org/articles/nunit/writing-tests/attributes/apartment.html" rel="noopener nofollow"><span class="md-plain">Apartment Attribute</span></a></span></span></td>
<td><span class="td-span"><span class="md-plain">指示测试应在特定单元中运行。</span></span></td>
</tr>
<tr class="md-end-block">
<td><span class="td-span"><span class="md-meta-i-c  md-link"><a href="https://docs.nunit.org/articles/nunit/writing-tests/attributes/author.html" rel="noopener nofollow"><span class="md-plain">Author Attribute</span></a></span></span></td>
<td><span class="td-span"><span class="md-plain">提供测试作者的姓名。</span></span></td>
</tr>
<tr class="md-end-block">
<td><span class="td-span"><span class="md-meta-i-c  md-link"><a href="https://docs.nunit.org/articles/nunit/writing-tests/attributes/category.html" rel="noopener nofollow"><span class="md-plain">Category Attribute</span></a></span></span></td>
<td><span class="td-span"><span class="md-plain">为测试指定一个或多个类别。</span></span></td>
</tr>
<tr class="md-end-block">
<td><span class="td-span"><span class="md-meta-i-c  md-link"><a href="https://docs.nunit.org/articles/nunit/writing-tests/attributes/combinatorial.html" rel="noopener nofollow"><span class="md-plain">Combinatorial Attribute</span></a></span></span></td>
<td><span class="td-span"><span class="md-plain">为提供的值的所有可能组合生成测试用例。</span></span></td>
</tr>
<tr class="md-end-block">
<td><span class="td-span"><span class="md-meta-i-c  md-link"><a href="https://docs.nunit.org/articles/nunit/writing-tests/attributes/culture.html" rel="noopener nofollow"><span class="md-plain">Culture Attribute</span></a></span></span></td>
<td><span class="td-span"><span class="md-plain">指定应为其运行测试或夹具的区域性。</span></span></td>
</tr>
<tr class="md-end-block">
<td><span class="td-span"><span class="md-meta-i-c  md-link"><a href="https://docs.nunit.org/articles/nunit/writing-tests/attributes/datapoint.html" rel="noopener nofollow"><span class="md-plain">Datapoint Attribute</span></a></span></span></td>
<td><span class="td-span"><span class="md-plain">提供数据<span class="md-meta-i-c  md-link"><a href="https://docs.nunit.org/articles/nunit/writing-tests/attributes/theory.html" rel="noopener nofollow"><span class="md-plain">理论</span></a><span class="md-plain">.</span></span></span></span></td>
</tr>
<tr class="md-end-block">
<td><span class="td-span"><span class="md-meta-i-c  md-link"><a href="https://docs.nunit.org/articles/nunit/writing-tests/attributes/datapointsource.html" rel="noopener nofollow"><span class="md-plain">DatapointSource Attribute</span></a></span></span></td>
<td><span class="td-span"><span class="md-plain">提供数据源<span class="md-meta-i-c  md-link"><a href="https://docs.nunit.org/articles/nunit/writing-tests/attributes/theory.html" rel="noopener nofollow"><span class="md-plain">理论</span></a><span class="md-plain">.</span></span></span></span></td>
</tr>
<tr class="md-end-block">
<td><span class="td-span"><span class="md-meta-i-c  md-link"><a href="https://docs.nunit.org/articles/nunit/writing-tests/attributes/defaultfloatingpointtolerance.html" rel="noopener nofollow"><span class="md-plain">DefaultFloatingPointTolerance Attribute</span></a></span></span></td>
<td><span class="td-span"><span class="md-plain">指示测试应使用指定的容差作为浮点数和双重比较的默认值。</span></span></td>
</tr>
<tr class="md-end-block">
<td><span class="td-span"><span class="md-meta-i-c  md-link"><a href="https://docs.nunit.org/articles/nunit/writing-tests/attributes/description.html" rel="noopener nofollow"><span class="md-plain">Description Attribute</span></a></span></span></td>
<td><span class="td-span"><span class="md-plain">将描述性文本应用于测试、测试修复或程序集。</span></span></td>
</tr>
<tr class="md-end-block">
<td><span class="td-span"><span class="md-meta-i-c  md-link"><a href="https://docs.nunit.org/articles/nunit/writing-tests/attributes/explicit.html" rel="noopener nofollow"><span class="md-plain">Explicit Attribute</span></a></span></span></td>
<td><span class="td-span"><span class="md-plain">指示除非显式运行，否则应跳过测试。</span></span></td>
</tr>
<tr class="md-end-block">
<td><span class="td-span"><span class="md-meta-i-c  md-link"><a href="https://docs.nunit.org/articles/nunit/writing-tests/attributes/fixturelifecycle.html" rel="noopener nofollow"><span class="md-plain">FixtureLifeCycle Attribute</span></a></span></span></td>
<td><span class="td-span"><span class="md-plain">指定夹具的生命周期，允许为每个测试用例构造测试夹具的新实例。在测试用例并行性很重要的情况下很有用。</span></span></td>
</tr>
<tr class="md-end-block">
<td><span class="td-span"><span class="md-meta-i-c  md-link"><a href="https://docs.nunit.org/articles/nunit/writing-tests/attributes/ignore.html" rel="noopener nofollow"><span class="md-plain">Ignore Attribute</span></a></span></span></td>
<td><span class="td-span"><span class="md-plain">指示由于某种原因不应运行测试。</span></span></td>
</tr>
<tr class="md-end-block">
<td><span class="td-span"><span class="md-meta-i-c  md-link"><a href="https://docs.nunit.org/articles/nunit/writing-tests/attributes/levelofparallelism.html" rel="noopener nofollow"><span class="md-plain">LevelOfParallelism Attribute</span></a></span></span></td>
<td><span class="td-span"><span class="md-plain">指定程序集级别的并行度级别。</span></span></td>
</tr>
<tr class="md-end-block">
<td><span class="td-span"><span class="md-meta-i-c  md-link"><a href="https://docs.nunit.org/articles/nunit/writing-tests/attributes/maxtime.html" rel="noopener nofollow"><span class="md-plain">MaxTime Attribute</span></a></span></span></td>
<td><span class="td-span"><span class="md-plain">指定测试用例成功的最长时间（以毫秒为单位）。</span></span></td>
</tr>
<tr class="md-end-block">
<td><span class="td-span"><span class="md-meta-i-c  md-link"><a href="https://docs.nunit.org/articles/nunit/writing-tests/attributes/nonparallelizable.html" rel="noopener nofollow"><span class="md-plain">NonParallelizable Attribute</span></a></span></span></td>
<td><span class="td-span"><span class="md-plain">指定测试及其后代不能并行运行。</span></span></td>
</tr>
<tr class="md-end-block">
<td><span class="td-span"><span class="md-meta-i-c  md-link"><a href="https://docs.nunit.org/articles/nunit/writing-tests/attributes/nontestassembly.html" rel="noopener nofollow"><span class="md-plain">NonTestAssembly Attribute</span></a></span></span></td>
<td><span class="td-span"><span class="md-plain">指定程序集引用 NUnit 框架，但不包含测试。</span></span></td>
</tr>
<tr class="md-end-block">
<td><span class="td-span"><span class="md-meta-i-c  md-link"><a href="https://docs.nunit.org/articles/nunit/writing-tests/attributes/onetimesetup.html" rel="noopener nofollow"><span class="md-plain">OneTimeSetUp Attribute</span></a></span></span></td>
<td><span class="td-span"><span class="md-plain">标识在任何子测试之前要调用一次的方法。</span></span></td>
</tr>
<tr class="md-end-block">
<td><span class="td-span"><span class="md-meta-i-c  md-link"><a href="https://docs.nunit.org/articles/nunit/writing-tests/attributes/onetimeteardown.html" rel="noopener nofollow"><span class="md-plain">OneTimeTearDown Attribute</span></a></span></span></td>
<td><span class="td-span"><span class="md-plain">标识在所有子测试之后要调用一次的方法。</span></span></td>
</tr>
<tr class="md-end-block">
<td><span class="td-span"><span class="md-meta-i-c  md-link"><a href="https://docs.nunit.org/articles/nunit/writing-tests/attributes/order.html" rel="noopener nofollow"><span class="md-plain">Order Attribute</span></a></span></span></td>
<td><span class="td-span"><span class="md-plain">指定在包含的夹具或套件中运行装饰测试的顺序。</span></span></td>
</tr>
<tr class="md-end-block">
<td><span class="td-span"><span class="md-meta-i-c  md-link"><a href="https://docs.nunit.org/articles/nunit/writing-tests/attributes/pairwise.html" rel="noopener nofollow"><span class="md-plain">Pairwise Attribute</span></a></span></span></td>
<td><span class="td-span"><span class="md-plain">为提供的所有可能值对生成测试用例。</span></span></td>
</tr>
<tr class="md-end-block">
<td><span class="td-span"><span class="md-meta-i-c  md-link"><a href="https://docs.nunit.org/articles/nunit/writing-tests/attributes/parallelizable.html" rel="noopener nofollow"><span class="md-plain">Parallelizable Attribute</span></a></span></span></td>
<td><span class="td-span"><span class="md-plain">指示测试和/或其后代是否可以并行运行。</span></span></td>
</tr>
<tr class="md-end-block">
<td><span class="td-span"><span class="md-meta-i-c  md-link"><a href="https://docs.nunit.org/articles/nunit/writing-tests/attributes/platform.html" rel="noopener nofollow"><span class="md-plain">Platform Attribute</span></a></span></span></td>
<td><span class="td-span"><span class="md-plain">指定应为其运行测试或夹具的平台。</span></span></td>
</tr>
<tr class="md-end-block">
<td><span class="td-span"><span class="md-meta-i-c  md-link"><a href="https://docs.nunit.org/articles/nunit/writing-tests/attributes/property.html" rel="noopener nofollow"><span class="md-plain">Property Attribute</span></a></span></span></td>
<td><span class="td-span"><span class="md-plain">允许在任何测试用例或夹具上设置命名属性。</span></span></td>
</tr>
<tr class="md-end-block">
<td><span class="td-span"><span class="md-meta-i-c  md-link"><a href="https://docs.nunit.org/articles/nunit/writing-tests/attributes/random.html" rel="noopener nofollow"><span class="md-plain">Random Attribute</span></a></span></span></td>
<td><span class="td-span"><span class="md-plain">指定生成随机值作为参数化测试的参数。</span></span></td>
</tr>
<tr class="md-end-block">
<td><span class="td-span"><span class="md-meta-i-c  md-link"><a href="https://docs.nunit.org/articles/nunit/writing-tests/attributes/range.html" rel="noopener nofollow"><span class="md-plain">Range Attribute</span></a></span></span></td>
<td><span class="td-span"><span class="md-plain">指定一系列值作为参数化测试的参数。</span></span></td>
</tr>
<tr class="md-end-block">
<td><span class="td-span"><span class="md-meta-i-c  md-link"><a href="https://docs.nunit.org/articles/nunit/writing-tests/attributes/repeat.html" rel="noopener nofollow"><span class="md-plain">Repeat Attribute</span></a></span></span></td>
<td><span class="td-span"><span class="md-plain">指定应多次执行修饰方法。</span></span></td>
</tr>
<tr class="md-end-block">
<td><span class="td-span"><span class="md-meta-i-c  md-link"><a href="https://docs.nunit.org/articles/nunit/writing-tests/attributes/requiresthread.html" rel="noopener nofollow"><span class="md-plain">RequiresThread Attribute</span></a></span></span></td>
<td><span class="td-span"><span class="md-plain">指示测试方法、类或程序集应在单独的线程上运行。</span></span></td>
</tr>
<tr class="md-end-block">
<td><span class="td-span"><span class="md-meta-i-c  md-link"><a href="https://docs.nunit.org/articles/nunit/writing-tests/attributes/retry.html" rel="noopener nofollow"><span class="md-plain">Retry Attribute</span></a></span></span></td>
<td><span class="td-span"><span class="md-plain">如果测试失败，则导致重新运行测试，最多可达最大次数。</span></span></td>
</tr>
<tr class="md-end-block">
<td><span class="td-span"><span class="md-meta-i-c  md-link"><a href="https://docs.nunit.org/articles/nunit/writing-tests/attributes/sequential.html" rel="noopener nofollow"><span class="md-plain">Sequential Attribute</span></a></span></span></td>
<td><span class="td-span"><span class="md-plain">使用按提供的顺序排列的值生成测试用例，无需其他组合。</span></span></td>
</tr>
<tr class="md-end-block">
<td><span class="td-span"><span class="md-meta-i-c  md-link"><a href="https://docs.nunit.org/articles/nunit/writing-tests/attributes/setculture.html" rel="noopener nofollow"><span class="md-plain">SetCulture Attribute</span></a></span></span></td>
<td><span class="td-span"><span class="md-plain">设置测试持续时间内的当前区域性。</span></span></td>
</tr>
<tr class="md-end-block">
<td><span class="td-span"><span class="md-meta-i-c  md-link"><a href="https://docs.nunit.org/articles/nunit/writing-tests/attributes/setuiculture.html" rel="noopener nofollow"><span class="md-plain">SetUICulture Attribute</span></a></span></span></td>
<td><span class="td-span"><span class="md-plain">设置测试持续时间内的当前 UI 区域性。</span></span></td>
</tr>
<tr class="md-end-block">
<td><span class="td-span"><span class="md-meta-i-c  md-link"><a href="https://docs.nunit.org/articles/nunit/writing-tests/attributes/setup.html" rel="noopener nofollow"><span class="md-plain">SetUp Attribute</span></a></span></span></td>
<td><span class="td-span"><span class="md-plain">指示在每个测试方法之前调用的 TestFixture 方法。</span></span></td>
</tr>
<tr class="md-end-block">
<td><span class="td-span"><span class="md-meta-i-c  md-link"><a href="https://docs.nunit.org/articles/nunit/writing-tests/attributes/setupfixture.html" rel="noopener nofollow"><span class="md-plain">SetUpFixture Attribute</span></a></span></span></td>
<td><span class="td-span"><span class="md-plain">使用命名空间中所有测试夹具的一次性设置或拆卸方法标记类。</span></span></td>
</tr>
<tr class="md-end-block">
<td><span class="td-span"><span class="md-meta-i-c  md-link"><a href="https://docs.nunit.org/articles/nunit/writing-tests/attributes/singlethreaded.html" rel="noopener nofollow"><span class="md-plain">SingleThreaded Attribute</span></a></span></span></td>
<td><span class="td-span"><span class="md-plain">标记要求其所有测试在同一线程上运行的夹具。</span></span></td>
</tr>
<tr class="md-end-block">
<td><span class="td-span"><span class="md-meta-i-c  md-link"><a href="https://docs.nunit.org/articles/nunit/writing-tests/attributes/teardown.html" rel="noopener nofollow"><span class="md-plain">TearDown Attribute</span></a></span></span></td>
<td><span class="td-span"><span class="md-plain">指示在每个测试方法之后调用的 TestFixture 方法。</span></span></td>
</tr>
<tr class="md-end-block">
<td><span class="td-span"><span class="md-meta-i-c  md-link"><a href="https://docs.nunit.org/articles/nunit/writing-tests/attributes/test.html" rel="noopener nofollow"><span class="md-plain">Test Attribute</span></a></span></span></td>
<td><span class="td-span"><span class="md-plain">标记表示测试的 TestFixture 的方法。</span></span></td>
</tr>
<tr class="md-end-block">
<td><span class="td-span"><span class="md-meta-i-c  md-link"><a href="https://docs.nunit.org/articles/nunit/writing-tests/attributes/testcase.html" rel="noopener nofollow"><span class="md-plain">TestCase Attribute</span></a></span></span></td>
<td><span class="td-span"><span class="md-plain">将带有参数的方法标记为测试，并提供内联参数。</span></span></td>
</tr>
<tr class="md-end-block">
<td><span class="td-span"><span class="md-meta-i-c  md-link"><a href="https://docs.nunit.org/articles/nunit/writing-tests/attributes/testcasesource.html" rel="noopener nofollow"><span class="md-plain">TestCaseSource Attribute</span></a></span></span></td>
<td><span class="td-span"><span class="md-plain">将带有参数的方法标记为测试，并提供参数源。</span></span></td>
</tr>
<tr class="md-end-block">
<td><span class="td-span"><span class="md-meta-i-c  md-link"><a href="https://docs.nunit.org/articles/nunit/writing-tests/attributes/testfixture.html" rel="noopener nofollow"><span class="md-plain">TestFixture Attribute</span></a></span></span></td>
<td><span class="td-span"><span class="md-plain">将类标记为测试夹具，并可能提供内联构造函数参数。</span></span></td>
</tr>
<tr class="md-end-block">
<td><span class="td-span"><span class="md-meta-i-c  md-link"><a href="https://docs.nunit.org/articles/nunit/writing-tests/attributes/testfixturesetup.html" rel="noopener nofollow"><span class="md-plain">TestFixtureSetup Attribute</span></a></span></span></td>
<td><span class="td-span"><span class="md-plain">已弃用，同义词： <span class="md-meta-i-c  md-link"><a href="https://docs.nunit.org/articles/nunit/writing-tests/attributes/onetimesetup.html" rel="noopener nofollow"><span class="md-plain">OneTimeSetUp Attribute</span></a><span class="md-plain">.</span></span></span></span></td>
</tr>
<tr class="md-end-block">
<td><span class="td-span"><span class="md-meta-i-c  md-link"><a href="https://docs.nunit.org/articles/nunit/writing-tests/attributes/testfixturesource.html" rel="noopener nofollow"><span class="md-plain">TestFixtureSource Attribute</span></a></span></span></td>
<td><span class="td-span"><span class="md-plain">将类标记为测试夹具，并为构造函数参数提供源。</span></span></td>
</tr>
<tr class="md-end-block">
<td><span class="td-span"><span class="md-meta-i-c  md-link"><a href="https://docs.nunit.org/articles/nunit/writing-tests/attributes/testfixtureteardown.html" rel="noopener nofollow"><span class="md-plain">TestFixtureTeardown Attribute</span></a></span></span></td>
<td><span class="td-span"><span class="md-plain">已弃用，同义词： <span class="md-meta-i-c  md-link"><a href="https://docs.nunit.org/articles/nunit/writing-tests/attributes/onetimeteardown.html" rel="noopener nofollow"><span class="md-plain">OneTimeTearDown Attribute</span></a><span class="md-plain">.</span></span></span></span></td>
</tr>
<tr class="md-end-block">
<td><span class="td-span"><span class="md-meta-i-c  md-link"><a href="https://docs.nunit.org/articles/nunit/writing-tests/attributes/testof.html" rel="noopener nofollow"><span class="md-plain">TestOf Attribute</span></a></span></span></td>
<td><span class="td-span"><span class="md-plain">指示要测试的类的名称或类型。</span></span></td>
</tr>
<tr class="md-end-block">
<td><span class="td-span"><span class="md-meta-i-c  md-link"><a href="https://docs.nunit.org/articles/nunit/writing-tests/attributes/theory.html" rel="noopener nofollow"><span class="md-plain">Theory Attribute</span></a></span></span></td>
<td><span class="td-span"><span class="md-plain">将测试方法标记为理论，这是NUnit中的一种特殊测试。</span></span></td>
</tr>
<tr class="md-end-block">
<td><span class="td-span"><span class="md-meta-i-c  md-link"><a href="https://docs.nunit.org/articles/nunit/writing-tests/attributes/timeout.html" rel="noopener nofollow"><span class="md-plain">Timeout Attribute</span></a></span></span></td>
<td><span class="td-span"><span class="md-plain">为测试用例提供超时值（以毫秒为单位）。</span></span></td>
</tr>
<tr class="md-end-block">
<td><span class="td-span"><span class="md-meta-i-c  md-link"><a href="https://docs.nunit.org/articles/nunit/writing-tests/attributes/values.html" rel="noopener nofollow"><span class="md-plain">Values Attribute</span></a></span></span></td>
<td><span class="td-span"><span class="md-plain">为测试方法的参数提供一组内联值。</span></span></td>
</tr>
<tr class="md-end-block">
<td><span class="td-span"><span class="md-meta-i-c  md-link"><a href="https://docs.nunit.org/articles/nunit/writing-tests/attributes/valuesource.html" rel="noopener nofollow"><span class="md-plain">ValueSource Attribute</span></a></span></span></td>
<td><span class="td-span"><span class="md-plain">为测试方法的参数提供值的源</span></span></td>
</tr>
</tbody>
</table></div>

新建一个类，创建一个测试方法，在其前面加上[Test]，代码中右键点击 运行测试（或按Ctrl+R,T），稍等片刻即可在测试资源管理器看到结果：

## 五、TestFixture

功能：此属性标记包含测试以及（可选）设置或拆解方法的类。

现在，对用作测试夹具的类的大多数限制都已消除。TestFixture类：

- 可以是公共的、受保护的、私有的或内部的。
- 可能是静态类。
- 可以是泛型的，只要提供了任何类型参数，或者可以从实际参数中推断出来。
- 可能不是抽象的 - 尽管该属性可以应用于旨在用作TestFixture基类的抽象类。
- 如果 TestFixtureAttribute 中没有提供任何参数，则该类必须具有默认构造函数。
- 如果提供了参数，则它们必须与其中一个构造函数匹配。

如果违反了这些限制中的任何一个，则该类不可作为测试运行，并且将显示为错误。

建议构造函数没有任何副作用，因为 NUnit 可能会在会话过程中多次构造对象。

从 NUnit 2.5 开始，TestFixture 属性对于非参数化、非通用Fixture是可选的。只要该类包含至少一个标有 Test、TestCase 或 TestCaseSource 属性的方法，它就会被视为TestFixture。

```csharp
using NUnit.Framework;

namespace MyTest;

// [TestFixture] // 2.5 版本以后，可选
public class FirstTest
{

    [Test]
    public void Test1()
    {
        Console.WriteLine("test1,hello");
    }
    
}
```

总结：在2.5版本的以前的nunit需要在测试类前加上 TestFixtureAttribute，2.5版本后可选。

## 六、SetUp 设置

功能：此属性在TestFixture内部使用，以提供在调用每个测试方法之前执行的一组通用函数，用于初始化单元测试的一些数据。（意味着假如在某个方法前加上SetUpAttribute，则在执行测试方法前会先执行这个方法）

SetUp 方法可以是静态方法，也可以是实例方法，您可以在夹具中定义多个方法。通常，多个 SetUp 方法仅在继承层次结构的不同级别定义，如下所述。

如果 SetUp 方法失败或引发异常，则不会执行测试，并报告失败或错误。

实践：运行以下测试，观察测试结果：

```csharp
using NUnit.Framework;

namespace ConsoleApp4
{
    // [TestFixture] // 2.5 版本以后，可选
    public class MyTestUnit
    {
        private string? Name;

        [SetUp]
        public void InitTest()
        {
            Console.WriteLine("初始化单元测试...");
            Name = "爱莉小跟班gaolx";
        }

        private int a = 10;
        [OneTimeSetUp] // 只执行一次
        public void OneTime()
        {
            a++;
            Console.WriteLine("我只执行一次");
        }

        [Test]
        public void Test01()
        {
            Console.WriteLine($"我的名字是{Name}");
            Console.WriteLine($"a的值是：{a}");
        }
    }
}

```

结果：

**继承**  

SetUp 属性继承自任何基类。因此，如果基类定义了 SetUp 方法，则会在派生类中的每个测试方法之前调用该方法。

您可以在基类中定义一个 SetUp 方法，在派生类中定义另一个方法。NUnit 将在派生类中调用基类 SetUp 方法之前调用基类 SetUp 方法。

**警告**
如果在派生类中重写了基类 SetUp 方法，则 NUnit 将不会调用基类 SetUp 方法;NUnit 预计不会使用包括隐藏基方法在内的用法。请注意，每种方法可能都有不同的名称;只要两者都存在属性，每个属性都将以正确的顺序调用。[SetUp]

**笔记**  

1. 尽管可以在同一类中定义多个 SetUp 方法，但您很少应该这样做。与在继承层次结构中的单独类中定义的方法不同，不能保证它们的执行顺序。
2. 在 .NET 4.0 或更高版本下运行时，如有必要，可以指定异步方法（c# 中的关键字）。
