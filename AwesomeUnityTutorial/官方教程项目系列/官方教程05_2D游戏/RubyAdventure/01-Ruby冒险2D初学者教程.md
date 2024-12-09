# Ruby's Adventure：2D 初学者
  
【百日挑战44】unity教程之2D游戏开发初步（八）  
前言：学完了官方项目《2DGameKit》的全部内容，复习完了前期unity的基础知识，我们开始学习一个新的教程系列《RubyAdventure2DRpg》，这是一个2D的RPG游戏，我们就开始更加深入的学习2D游戏开发中的各个细节，今天我们先只讲Sprite的导入和Sprite的移动。
> 资源链接：
>
> - [中文官方教程地址](https://learn.unity.com/project/ruby-s-adventure-2d-chu-xue-zhe)
> - [AssetStore](https://assetstore.unity.com/packages/templates/tutorials/2d-beginner-tutorial-resources-140167?_ga=2.134705203.331241089.1633678521-522971275.1624332126)
> - [Baidu 云盘](https://pan.baidu.com/s/193a7getqsQewRm16UIzqmg) 提取码: iw1x
> - [迅雷云盘](https://pan.xunlei.com/s/VMleChfhqqDLB9q7E3OvLfTWA1) 提取码：d3ed
  
## 开始之前01：导入Package
  
在学习我们这门课程之前当然需要先准备好资源，一般在Asset Store里面搜索《2D Beginner: Tutorial Resources》按照之前的方式添加到我的资源再到unity用Package Manager下载并导入Package所有文件到一个空项目即可。
  
如果网络不好的也可以先把unitypackage下载到本地再手动导入
> 本地 unitypackage 文件使用：  
> 新建项目 --> 打开项目 --> 将 unitypackage 文件拖入已打开 unity 的界面的 Project 窗口中 --> 选择需要导入的资源，import 导入
  
如果导入之后没有报错，则在Asset/Scene 路径下新建一个场景 MainScene。方法如下：  
1.选择 File > New Scene。或者，可以使用 Ctrl + N (Windows/Linux) 或 Cmd + N (macOS) 快捷键。  
2.如果显示一个弹窗提示你还有未保存的更改，这是因为你移动或更改了演示场景中的某个对象，所以 Unity 要确认你是否打算放弃这些更改。只需单击 Don’t Save。  
3.现在你已创建名为“Untitled”的空场景。这意味着尚未将这个场景写入磁盘。  
  
接下来，使用恰当名称来保存此场景。选择 File > Save 或者使用 Ctrl/Cmd + S 快捷键。  
  
4.选择 Scenes 文件夹，并将场景命名为“MainScene”。  
  
现在你有了一个可以使用的场景。在余下的教程中，你将使用此场景。请记住要经常按 Ctrl/Cmd + S，将你的更改保存到磁盘。这样，如果你退出 Unity 后再返回，就不会丢失已进行的更改。
  
## 开始之前02：导入Sprite
  
试着将教程中的一张图片导入项目的Assets里面，可以下载之后直接拖入Assets，也可以另存为图片到我们的项目中。
  
可以看到，我们导入的png图片格式在unity中被重新被导入为了Sprite（精灵），下面就是资产导入的各项设置

## 1. 项目简介

2D RPG 游戏制作教程，特点：

- 学习使用 2D 资源
- 学习简单的 2D RPG 游戏制作流程
- 只提供素材（不同于前面的官方教程，提供完整的源码，预制件等，需要自己思考如何将这些素材通过代码组装在一起）
- 学习创建并控制角色（使用脚本代码）
- 学习使用瓦片地图创建世界
- 学习设置动态精灵（Sprite）
- 学习一些简单特效（粒子效果）

## 2. 创建并控制角色

### 2.1 创建角色

#### 2.1.1 使用静态精灵创建角色

我们希望将Ruby作为我们的游戏角色放置在Scene中，不需要具体的定位就拖拽到hierachy中，默认位置(0,0,0)
  
精灵 Sprite 是 Unity 中 2D 素材的默认存在形式，是 Unity 中的 2D 图形对象。

在 2D 游戏中，使用 2D 素材的转换过程：  
PNG（JPG 等）----> Sprite ----> GameObject
  
观察我们从Asset中拖进来的这个sprite，里面除了Transform组件，还包括了一个Sprite Renderer 组件，负责将效果呈现在屏幕上的一个组件，没有这个组件，精美的图像将无法显示，关于该组件各选项的作用如下：

### 2.2 移动角色

#### 2.2.1 移动精灵

在讲unity基础知识的时候我们提到过笛卡尔坐标系，2D 游戏中，没有 Z 轴来表示深度，只有 X 轴和 Y 轴，以中心点为原点（0,0），左 x 负，右 x 正，上 y 正，下 y 负，要想表示远近只能用图层排序——谁遮住谁

![](../../../imgs/2d坐标.png)

Unity 中，通过游戏对象的 Transform 组件，可以获取该对象在场景中的位置 Position，并通过更改 Transform 组件 Position 的值(x, y)，可以更改其位置，依据的坐标轴就是上面描述的 2D 坐标轴
  
例如这里将x轴的位置设置为-2，可以看到Sprite向左平移了两个单位
  
简单的介绍下Sprite导入、Sprite Renderer和游戏对象的移动，下期我们将讲解如何用c#代码移动Sprite。
  
【百日挑战45】unity教程之2D游戏开发初步（九）  
  
前言：在上期教程中，我们通过官方一个新的2D的RPG游戏教程系列《RubyAdventure2DRpg》学习了场景的创建、Sprite导入、Sprite Renderer和游戏对象的移动，今天我们继续讲解如何用c#代码来控制Sprite的移动。
  
#### 2.2.2 Vector2 二维向量

在数学中，Vector 向量/矢量指的是带方向的线段

在 Unity 中，Transform 值使用 x 表示水平位置，使用 y 表示垂直位置，使用 z 表示深度。这 3 个数值组成一个坐标。由于此游戏是 2D 游戏，你无需存储 z 轴位置，因此你可以在此处使用 Vector2 来仅存储 x 和 y 位置。

例如，Transform 中 position 的类型，也是 Vector2

C# 这种强类型语言，赋值时，左右必须是同一类型才能进行赋值
  
#### 2.2.3 用脚本移动精灵

在游戏对象Ruby中，点击“Add Component”，New Script新建脚本，类名RubyController，在 Update 方法中，更改 Ruby 角色位置
  
脚本默认两个事件函数说明：  
1、Update方法：该方法在游戏执行第一帧更新之前调用，每更新一帧都会执行一次。  
2、Start方法：该方法在每帧调用一次更新，游戏开始前只执行一次。
  
Unity 在每帧执行 Update 内的代码，为了形成动感，游戏（就像电影一样）是高速显示的静止图像。在游戏中通常会在一秒内显示 30 或 60 张图像。其中的一张图像称为一帧。

在此 Update 函数中，你可以编写想要在游戏中连续发生的任何操作（例如，读取玩家的输入、移动游戏对象或计算经过的时间）。

代码版本 1：

```C#
public class RubyController : MonoBehaviour
{
    // 每帧调用一次 Update
    // 让游戏对象每帧右移 0.1个单位(每一个方格代表1个单位)
    void Update()
    {
        // 创建一个 Vector2 对象 position，用来获取当前对象的位置
        Vector2 position = transform.position;
        // 更改 position 的 x 坐标值，让其 加上 0.1
        position.x = position.x + 0.1f;
        // 更新当前对象的位置到新位置
        transform.position = position;
    }
}
```
  
注：Vector2是UnityEngine自带的一个系统类，用于描述二维向量的位置（x,y坐标值），transform 获取当前挂载了改脚本组件地游戏对象的transform对象值，包括位移，旋转，缩放等，后面的.position就是获取当前对象的，然后复制给这个Vector2向量，0.1f中的f是一个关键字，float的缩写，小数常量后面加一个f表示单精度浮点数，不加f则默认为双精度浮点数
  
保存运行，可以看到Ruby不断地快速的向右移动。
  

### 2.3 角色控制器与键盘输入

控制角色，是游戏中最基本的用户交互，上期教程我们通过脚本实现了Ruby每帧沿着x轴右移动0.1个单位，我们将试着通过键盘输入控制角色移动

#### 2.3.1 游戏中的控制方式

根据输入硬件的不同，控制方式可以有以下几种：

- 鼠标键盘（PC端为主）
- 手机触屏、重力（通过陀螺仪等传感器，移动端为主）
- 手柄（主机平台如 PS5、Xbox，掌机平台如Switch）
- 体感（街机、部分主机端）
- 可穿戴设备，比如 VR 、AR 眼镜 常用的瞳孔控制
- 声音控制

考虑到多平台适配问题，在实际项目中，需要一套相应的框架判断发布的目标平台选择对应的输入方式
  
讲完了二维向量及在游戏开发中的应用，通过脚本实现了Ruby每帧沿着x轴右移，科普了常见的几种输入方式，下期教程中，我们将详细讲解用键盘控制代码来操作我们角色的移动
  
【百日挑战46】unity教程之2D游戏开发初步（十）  
  
前言：在上期教程中，我们通过官方一个新的2D的RPG游戏教程系列《RubyAdventure2DRpg》学习了二维向量及在游戏开发中的应用，通过脚本实现了Ruby每帧沿着x轴右移，科普了常见的几种输入方式，今天我们接着学习如何用代码实现键盘控制
  
#### 2.3.2 键盘控制代码

这里我们用最原始的键盘控制角色，将下列代码复制到vs中，按键盘上的 Horizontal 和 Vertical 键，inputsystem可根据按键的不同获取不同的值。
  
input 是UnityEngine中用于管理用户输入的的一个类，GetAxis 是获取轴，轴的定义根据inputsettings配置文件将轴与实际的按键一一映射
  
代码版本 2：
  
```C#
public class RubyController : MonoBehaviour
{
   // 每帧调用一次 Update
    // 让游戏对象每帧右移 0.1
    void Update()
    {
        // 获取水平输入，按向左，会获得 -1.0 f ; 按向右，会获得 1.0 f
        float horizontal = Input.GetAxis("Horizontal");
        // 获取垂直输入，按向下，会获得 -1.0 f ; 按向上，会获得 1.0 f
        float vertical = Input.GetAxis("Vertical");

        // 获取对象当前位置
        // 二维向量position = 物体当前位置 + 移动位置(每帧执行update方法都会执行一次) + 移动的方向
        Vector2 position = transform.position;
        // 响应操作，更改位置
        position.x = position.x + 0.1f * horizontal;
        position.y = position.y + 0.1f * vertical;
        // 新位置给游戏对象
        transform.position = position;
    }
}

```
  
保存运行，按键盘上的wasd键或者上下左右，可以看到Ruby会沿着键盘按下按键相应的方向自由移动。
  
#### 2.3.3 Unity 默认 Input Manager 设置

在 Unity 项目设置中，可以通过 Input Manager 进行默认的游戏输入控制设置（新版的unity升级为了Input System，可在Player设置里切换）
  
Edit > Project Settings > Input
  
关于 Input Manager：
输入管理器窗口允许您为项目定义输入轴及其相关操作。要访问它，请从 Unity 的主菜单转到“编辑”>“项目设置”，然后从右侧导航中选择“输入管理器” 。
  
输入管理器使用以下类型的控件：
· 键是指物理键盘上的任意键，例如 W、Shift 或空格键。
· 按钮是指物理控制器（例如游戏手柄）上的任何按钮，例如遥控器上的 X 按钮。
· 虚拟轴（复数：axes）映射到控件，例如按钮或按键。当用户激活控件时，轴接收 [–1..1] 范围内的值。
  
键盘按键，以 2 个键来定义轴，left对应键盘的左键，right对应右键：
  
- 负值键 negative button，被按下时将轴设置为 -1
- 正值键 positive button ，被按下时将轴设置为 1
  
Alt negative button/Alt positive button 是可以代替的按键，效果与negative button/positive button是等同的，你也可以在这里更改其他的按键
  
Axis 轴 Axes 是它的负数形式
  
- Horizontal Axis： 水平轴 对应 X 轴
- Vertical Axis：纵轴 对应 Y 轴
  
#### 2.3.4 Input 类

[ UnityEngine.Input 官方 API 文档](https://docs.unity3d.com/cn/current/ScriptReference/Input.html)

使用该类来读取传统游戏输入中设置的轴/鼠标/按键，以及访问移动设备上的多点触控/加速度计数据。
  
若要使用输入来进行任何类型的移动行为，请使用 Input.GetAxis。 它为您提供平滑且可配置的输入 - 可以映射到键盘、游戏杆或鼠标。 请将 Input.GetButton 仅用于事件等操作（比如按键）。不要将它用于移动操作。Input.GetAxis 将使脚本代码更简洁。
  
复制粘贴以下的代码，尽管效果是等价于版本2的，但是写死在代码里面就无法随意在Input Manager更改键位映射了，会给后期的开发带来很大的麻烦，所以不推荐。
  
代码版本 3：

```C#
public class RubyController : MonoBehaviour
{
   // 每帧调用一次 Update
   // 可以这样做，但不建议
   void Update()
   {
       Vector2 position = transform.position;

       if(Input.GetKey("d")){
           position.x = position.x + 0.1f;
       }
       if(Input.GetKey("a")){
           position.x = position.x - 0.1f;
       }
       if(Input.GetKey("s")){
           position.y = position.y - 0.1f;
       }
       if(Input.GetKey("w")){
           position.y = position.y + 0.1f;
       }

       transform.position = position;
   }
}

```
  
至此，我们就详细的介绍完了如何使用键盘输入来控制我们的2D角色，细致的学习Unity中非常重要的输入模块——Input Manager，包括将其进行默认的游戏输入控制设置以及在代码中调用它操纵我们角色自由移动。
  
【百日挑战47】unity教程之2D游戏开发初步（十一）
  
前言：在上期教程中，我们通过官方一个新的2D的RPG游戏教程系列《RubyAdventure2DRpg》学习了如何使用键盘输入来控制我们的2D角色，细致的学习Unity中非常重要的输入模块——Input Manager，包括将其进行默认的游戏输入控制设置以及在代码中调用它操纵我们角色自由移动，今天我们接着学习关于游戏中的时间和帧率，以及改进之前角色控制器移动部分的弊端
  
#### 2.3.5 时间和帧率

上期我们完成了角色移动的代码，但是有个严重的问题，角色移动的速度太快，而且会随帧数的变化而不同，这显然是比较严重的问题，尽管我们限制每执行一次Update方法角色只移动0.1个单位，但是由于Update方法是每帧都会执行一次的，当前的代码中，帧数越高，同一时间内，执行 Update 的次数越多，角色移动速度越快。
  
Game视图中点击右上角“State”，即可查看游戏运行时的各种状态，其中“FPS”代表游戏运行时候的渲染帧数。
  
那什么是帧数的概念，怎么理解它呢？如果游戏以每秒 60 帧的速度运行，那么 Ruby 将移动 0.1 _ 60，因此每秒移动 6 个单位。但是，如果游戏以每秒 10 帧的速度运行，就像刚刚让游戏运行的那样，那么 Ruby 仅移动 0.1 _ 10，因此每秒移动 1 个单位！
  
如果一个玩家的计算机非常陈旧，只能以每秒 30 帧的速度运行游戏，而另一个玩家的计算机能以每秒 120 帧的速度运行游戏，那么这两个玩家的主角的移动速度会有很大差异。这样就会使游戏的难易程度提高或降低，具体取决于运行游戏的计算机。
  
而帧数是由硬件水平影响的（越好越高），不同电脑中，会导致游戏效果完全不同，我们希望一款游戏在硬件不同的设备上运行效果是相同的，一种是锁帧，设定游戏的最高帧数，一种是以单位/秒来表示 Ruby 的移动速度。
  
对此，你可以锁帧，但并不推荐
  
代码版本 4，重点关注Start方法：

```C#
public class RubyController : MonoBehaviour
{
    // 在第一次帧更新之前调用 Start
    void Start()
    {
        // 只有将垂直同步计数设置为0，才能锁帧，否则锁帧的代码无效
        // 垂直同步的作用就是显著减少游戏画面撕裂、跳帧，因为画面的渲染不是整个画面一同渲染的，而是逐行或逐列渲染的，能够让FPS保持与显示屏的刷新率相同。
        QualitySettings.vSyncCount = 0;
        //设定应用程序帧数为10
        Application.targetFrameRate = 10;
    }

    // 每帧调用一次 Update
    void Update()
    {
        float horizontal = Input.GetAxis("Horizontal");
        float vertical = Input.GetAxis("Vertical");

        Vector2 position = transform.position;
        position.x = position.x + 0.1f * horizontal;
        position.y = position.y + 0.1f * vertical;
        transform.position = position;
    }
}
```
  
保存运行，可以看到此时Ruby的移动速度就已经很慢了，而且帧数稳定在10帧，但是有点卡。
  
注：QualitySettings变量是游戏质量设置，你也可以将他们放到一个单独的QualityManager脚本内统一管理他们
  
> 相关资料：
>
> - [质量设置.垂直同步设置 官方 API](https://docs.unity3d.com/ScriptReference/QualitySettings-vSyncCount.html)
> - [垂直同步什么意思？游戏中垂直同步的作用](https://zhuanlan.zhihu.com/p/67370953)
  
关于垂直同步（V-SYNC）：  
· 垂直同步就是单个象素组成了水平扫描线，水平扫描线在垂直方向的堆积形成了完整的画面。垂直同步是一种让显示器延迟输入画面等待显卡渲染。
· 垂直同步的作用就是显著减少游戏画面撕裂、跳帧，因为画面的渲染不是整个画面一同渲染的，而是逐行或逐列渲染的，能够让FPS保持与显示屏的刷新率相同。  
· 例如，我们正常的显示器刷新率为60Hz，开启了垂直同步之后会将游戏锁帧，与显示器刷新率同步，主要用于显示器跟不上显卡输出的帧数能力（游戏帧数太高，而显示器达不到），而出现画面撕裂。
  
优点：使帧数平稳，防止帧数过高撕裂  
缺点：硬件能达到的情况下（显卡、显示器），锁帧会降低画面效果，不能充分利用硬件资源，而且会带来一定的操作延迟。

显然这样的作法太粗暴了，要解决此问题，你需要以单位/秒来表示 Ruby 的移动速度，而不是采用单位/帧（目前采用此设置）。锁帧本质上是解决帧数过高而造成画面撕裂的，并不是用来控制执行次数的。

为此，你需要通过将移动速度乘以 Unity 渲染一帧所需的时间来更改移动速度。如果游戏以每秒 10 帧的速度运行，则每帧耗时 0.1 秒。如果游戏以每秒 60 帧的速度运行，则每帧耗时 0.017 秒。如果将移动速度乘以该时间值，则移动速度将以秒表示。
  
复制粘贴以下代码，运行观察效果，可以看到尽管帧数很高，但角色移动速度依旧是恒定的，速度只受到speed变量影响了，不随帧数变化，而且可以在inspector中很方便的控制Ruby移动的速度，并做到了非常精确的速度控制。
  
**代码版本 5**

```C#
public class RubyController : MonoBehaviour
{
    // 将速度暴露出来，使其可调
    public float speed=0.1f;
// 每帧调用一次 Update
   void Update()
   {
       float horizontal = Input.GetAxis("Horizontal");
       float vertical = Input.GetAxis("Vertical");
       Vector2 position = transform.position;
       position.x = position.x + speed * horizontal * Time.deltaTime;
       position.y = position.y + speed * vertical * Time.deltaTime;
       transform.position = position;
   }
}
```
  
这里需要引入一个新的概念：Time.deltaTime 每帧的时间间隔，float 类型，从最后一帧到当前帧的间隔（以秒为单位）。用 1/当前的帧数 得到相应的时间，再结合Update，可得到一个恒定的变量，而且不受帧数影响
  
一般将这个值，用在 Update 方法中，乘以移动的距离（或角度），用来获取恒定（不同硬件水平的电脑间）的速度
  
至此，我们就学完了关于游戏中的时间和帧率，以及使用 Time.deltaTime 方法以单位/秒来表示 Ruby 的移动速度改进之前角色控制器移动部分随帧数不同移动速度不同的弊端，且做到不浪费硬件资源。

> 参考资料：
>
> - [Time.deltaTime 官方 API](https://docs.unity3d.com/ScriptReference/Time-deltaTime.html)
  
【百日挑战48】unity教程之2D游戏开发初步（十二）
  
前言：在上期教程中，我们通过官方一个新的2D的RPG游戏教程系列《RubyAdventure2DRpg》学习了关于游戏中的时间和帧率，以及使用 Time.deltaTime 方法以单位/秒来表示 Ruby 的移动速度改进之前角色控制器移动部分随帧数不同移动速度不同的弊端，且做到不浪费硬件资源。今天我们接着系统学习如何使用瓦片地图来创建我们的游戏世界。

</br>

</hr>
</br>

配套视频教程：
[https://space.bilibili.com/43644141/channel/seriesdetail?sid=299912](https://space.bilibili.com/43644141/channel/seriesdetail?sid=299912)

文章也同时同步微信公众号，喜欢使用手机观看文章的可以关注

![](../../imgs/微信公众号二维码.jpg)
