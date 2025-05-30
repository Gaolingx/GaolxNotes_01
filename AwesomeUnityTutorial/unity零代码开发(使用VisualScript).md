# Visual Script

> - [Visual Script 官方文档](https://docs.unity3d.com/Packages/com.unity.visualscripting@1.7/manual/index.html)
> - [unity 可视化开发官方专题](https://unity.com/cn/products/unity-visual-scripting)
> - [unity 可视化编程官方教程](https://learn.unity.com/project/introduction-to-visual-scripting)
  
【百日挑战23】unity教程之零代码开发游戏（一）
## 1. 简介

- 概念：
  可视化，零代码编程插件。  
   · 源自于 Bolt 插件，被 unity 收购后，改名为 Visual Script，可以使用拖拽的方式连接各个功能模块，构成一个完整的脚本，并完成一系列操作。
   · 可视化脚本使您无需编写代码即可为游戏或应用程序创建逻辑。  
   · 可视化脚本具有可视化的、基于单元的图形，程序员和非程序员都可以使用它们来设计最终逻辑或快速创建原型，往往会有事半功倍的效果。  
   · 可视化脚本基于使用图形元素的概念，图形元素代表函数、运算符或变量，并通过使用边从其端口连接单元。您不必逐行编写代码，您可以直观地完成所有工作。

- 安装：  
   从版本 2021.1 开始，默认情况下可视化脚本作为包安装。您可以在包管理器中查看它。  
   对于早期版本的 Unity, 2019 和 2020 LTS 版本，可以从 Unity Asset Store 免费下载 Bolt 可视化脚本。

![](/imgs/unity_visualScript.png)

## 2. 优缺点

- 优点：不用写代码，直观，用线构成流程，不逐行编写代码也能完成许多工作
- 缺点：会代码的，会觉得很麻烦，不如写代码快捷；而且，即使使用可视化编程，那些该学的编程理念一点也不能少，所以，有那个学习时间，代码一并也都学会了。不过，这只是我这个老程序员的看法，仁者见仁智者见智，存在的一定是合理的。

> 个人理解：
>
> - 可以辅助美术师和设计师自行测试素材
> - 可以进行儿童编程教学，也可以为一些对编程不是特别了解的人但又需要接触程序的人群使用（美术，策划...）
> - 可以辅助脚本代码使用，更底层的代码用脚本封装成函数，保证可维护性的前提让程序的执行逻辑一目了然
  
所以，哪怕是图形化编程也没有想象的那么简单，学习可视化编程不是意味着不用学习编程的概念了，不论是用脚本编程还是可视化编程，他们用到的算法，思路，概念还是基本一致的，只是编程的工具变了，从写代码变成了“连连看”，所以希望大家还是要沉下心，脚踏实地学习c#脚本，这是程序员的基础，并同时使用可视化编程为我们学习复杂脚本编程的铺路石，让我们能更好的理解编程逻辑。
说了这么多，接下来我们自己动手实操下如何使用Visual Script编写代码，还是以我们之前写的那段打印控制台的脚本为例。先来看看原代码
  
1、先删掉之前创建的MainPlayer组件，或者去掉组件左边的复选框，表示组件无效，代码自然也不会执行。
  
2、选中Player组件，添加Script Machine组件，在2020.03里面，从Asset Store 添加 Bolt后，添加组件 flow machine即可进行可视化编辑
  
3、在Script Machine组件中，脚本机和状态机的源（Source）有两个选项：图形文件 ( Graph ) 或嵌入固定资产（嵌入）。要在多个游戏对象中使用相同的图表，请使用图表源类型。您可能会遇到一些情况嵌入ded 图效果最好。
  
4、如果我们选择了“Graph”，则要给我们的Script Machine组件绑定一个Graph（图表），图形是一种可视化脚本资产，包含应用程序中逻辑的可视化表示。我们只要在Asset/Scripts里面新建一个“脚本图”即可，然后选中新建的脚本图，双击或点击“Edit Graph”，编辑我们的图表
  
5、进入编辑页面之后，你可以用鼠标右键搜索需要的方法，鼠标滚轮上下滚动视图，鼠标中键拖动视图，他们都会以节点的形式呈现，这里我们先添加一个字符串String Literal节点，值填上“我的名字是：
  
6、接下来继续添加变量，添加一个Get Variable节点，Kind选择Object（对象），变量这里填写myName，等同于 public string myName=“xxx”，下面的 this 表示当前对象。
  
7、根据打印的内容，由于我们需要组合变量和常量，这里需要用到 Add Inputs 节点，然后分别将变量和常量两个节点连在Add Inputs 节点的A，B上，表示A+B，如果你要加更多的量可以增加该节点的数字，这段共同组成了一个新的字符串（参数）："我的名字是：" + 爱莉小跟班。这些节点都是可以被重复使用的
  
8、添加一个事件函数节点On Start，表示在游戏开始的时候会执行这些代码，再添加Debug:Log(message)节点，表示将前面的字符串打印在控制台上，最后将他们连起来即可。
  
9、最后回到Player对象，找到Variable组件，添加name（变量名字）：myName，Type（类型）填：String，Value（值）填：爱莉小跟班，不要忘了将我们写的Graph绑定给Script Machine的Graph，保存运行，效果与之前一样。
  
所以，哪怕是图形化编程也没有想象的那么简单，学习可视化编程不是意味着不用学习编程的概念了，不论是用脚本编程还是可视化编程，他们用到的算法，思路，概念还是基本一致的，只是编程的工具变了，从写代码变成了“连连看”，所以希望大家还是要沉下心，脚踏实地学习c#脚本，这是程序员的基础，并同时使用可视化编程为我们学习复杂脚本编程的铺路石，让我们能更好的理解编程逻辑。
</br>
</hr>
</br>

配套视频教程：
[https://space.bilibili.com/43644141/channel/seriesdetail?sid=299912](https://space.bilibili.com/43644141/channel/seriesdetail?sid=299912)

文章也同时同步微信公众号，喜欢使用手机观看文章的可以关注

![](/imgs/微信公众号二维码.jpg)
