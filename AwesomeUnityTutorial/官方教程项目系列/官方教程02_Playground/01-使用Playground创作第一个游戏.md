# 使用 Playground 创作第一个游戏

> - [Playground 项目 unity 论坛链接](https://forum.unity.com/threads/unity-playground-official-thread.609982/)
> - [Playground 官方教程链接](https://learn.unity.com/project/unity-playground?language=en&courseId=5d532306edbc2a1334dd9aa8)
> - [项目资源及安装](../官方教程01_Unity软件界面介绍/01-安装教程所需资源.md)

Unity Playground 提供了一系列易于使用和整合的单任务组件，同时封装了开发过程中需要用到的各个功能模块，需要已经是一个比较完善的工程了，因此不要求具有
编程技能。通过将这些组件整合在一起，便可以创作出多种游戏风格的 2D 物理游戏。还可以使用 Playground 来了解游戏设计或关卡设计。
通过今天这节，我们将简单了解unity开发相关概念，以及一个游戏的制作流程，降低初学者学习曲线，减少后续学习的难度。

你将学到以下内容：
1、使用Playground制作第一个小游戏
2、了解常见的资产类型，知道各类资产在开发过程中的不同作用。
3、介绍Playground功能，他能够制作什么样的游戏
4、在制作2D小游戏的过程中，梳理我们会遇到的常见概念以及各种技术名词。

> 注意：  
> 官方文档中，使用的是 Unity 2017 ,版本比较旧，跟新的 Unity 界面有所区别，建议用新版本的同学，还是按此文档来操作，避免出现问题

在开始之前，我们要把官方的这个playground 包导入到我们的空项目中，这里首先

## 1. 制作你的第一个游戏

制作教程中第一个小游戏：飞船躲陨石——一个飞机在空中躲陨石吃小星星的赚分数的小游戏

### 1.1 创建玩家

1. 打开项目：  
   打开前面课程中已经准备好的 playground 项目
2. 创建新场景：  
进入编辑器后，可以看到unity自动为我们创建了个空的场景，不过这里用不到，我们这里先尝试学习手动创建一个新场景。
   project 窗口 --> Assets/scenes 文件夹（以后项目中所有的场景文件都放这里面，其他类型的文件同理，方便其他人查看我们的项目），右键 create scene ，创建只带有一个 camera 对象的场景，重命名为 PlaneAsteroid（命名可以根据自己喜好定义，最好是用字母、下划线、数字，慎用中文、空格）
3. 添加玩家飞船：
   - project 窗口 --> Assets/Images/SpaceShips/ 拖拽其中一个飞机到 Hierarchy 中（或者Scene视图中），让飞船对象成为 PlaneAsteroid scene 的子对象；

   显然飞船大小不合适，太大了，前面提到过scene 视图的缩放工具，或者在Inspector中直接调整Transform的Scale（缩放）来更改物体的大小。

   注：如果需要优先确定gameobject位置建议直接从asset拖进scene，如果需要优先确定物体的层级关系可以先拖进Hierarchy再到scene中定位
   - 选中飞船，在 inspector 窗口中，将对象命名为 Ship ，并将 Tag 选中为 Player

然后更改我们飞船的名字让他看上去更规范，例如Player01_Asteroid_Plane_Red，关于资产命名规则后期会有详细的介绍。
 
安排好飞船的位置之后，我们需要在inspector窗口的tag中给我们的飞船绑定 Player 的标签，这与脚本内设定的逻辑密切相关，包括碰陨石，吃星星，血量，死亡，胜负判定等等，有兴趣的可以自行研究script中脚本的相关代码
相关概念：
1.Scene（场景）：类似与我们演出用的舞台，gameobject所有的活动都会在此处进行。用于选择和定位景物、角色、摄像机、光源和所有其他类型的游戏对象。
2.Player（玩家）：

接下来我们快速介绍下move组件中的功能：

· “Type of  Control”指的是控制飞船移动所用的按键，“Arrow Keys”指的是键盘的四个方向键，可以改为“WASD”，这样操作会更加习惯。

· 速度（speed）：可以控制飞船移动的快慢，类似与玩游戏时候的“灵敏度”概念，play之后根据实际情况调整。

· 运动类型（Movement Type）：“All Direction”指的是飞船可以往任意方向移动，“Only Horizontal”和“Only Vertical”分别代表仅水平或仅垂直方向移动。

· Orient to direction：根据方向转向。

> 注意：  
> 尽可能拖拽素材到 Hierarchy 中，可以直接确定对象间父子关系；当然，如果为了定位方便，将素材拖拽到 scene 中，也可以，但记住要调整对象嵌套关系

4. 让飞船动起来：
后期大家是熟练之后可以使用input system响应玩家输入，这里为了降低难度，脚本当中已经为我们配置了输入，移动相关的代码。
   - 添加组件，将脚本：Move with Arrows 绑定到飞船对象上，这里可以直接把脚本拖进inspector窗口，也可以点击inspector窗口中的“Add Component”
   - 一旦你添加了 Move 脚本，还会自动添加一个 Rigidbody2D 组件。这是因为 Move 需要 Rigidbody2D 才能起作用。
   - 调整 Gizmos 中的脚本图标大小，将其缩小：使用 Scene 视图中的下拉表单辅助图标进行缩小。将 3D Icons 滑动条向左拖动，直到图标大小合适。

![](../../imgs/unity_3Dicon.png)

> 相关概念：
>  component（组件）：组件是每个游戏对象的功能部分。 不同的组件为游戏提供了相应的功能，组件包含您可以编辑的属性来定义游戏对象的行为。通过写c#脚本可以定制我们需要的功能。
> - tag（标签）： 利用标签，我们可以将对象分为几类，以便脚本仅在碰触到正确的对象时才能执行操作。没有标签，就无法区分对象。可以在项目配置中，增减标签
> - script（脚本）： Unity 中的游戏代码
> - sprite(精灵)： Unity 中的 2D 游戏对象素材

### 1.2 调整物理值

RigidBody2D 组件和移动脚本组件上，暴露出很多物理引擎相关的参数值，通过调整，可以改善游戏体验

- 在 RigidBody2D 组件上，我们要将 Gravity 修改为 0,否则物体一开始游戏就会往下掉。
- Orient to direction：船头朝向移动方向
- Friction ：摩擦力，设置为 5 可以消除漂移
- Speed ： 更改速度
- mass ：质量，影响惯性

如果对这些数据没有概念也不要紧，点击play按钮，在运行的时候调整这些组件的值，看看怎么设置更习惯。
> 注意：  
> 如果在运行模式下编辑组件值，则在游戏停止后，你所做的更改将丢失。切记要在非运行模式下进行更改！仅当测试某些即使丢失也没关系的临时值时，才应该在运行模式下进行更改。

![](../../imgs/unity_PhyNum.png)

> 相关概念：
>
> - 物理引擎：  
>   Unity 中，包含了完整的物理引擎，可以让开发者非常方便地模拟显示中的物理现象，比如：质量、惯性、重力、速度、加速度、碰撞、反弹等等。  
>   [物理系统 官方文档](https://docs.unity3d.com/cn/2021.1/Manual/PhysicsSection.html)

> - RigidBody（刚体）：  
>   是实现游戏对象的物理行为的主要组件。如果想让一个游戏对像能被物理引擎所影响，就必须为其添加 RigidBody 组件，分为 2D 和 3D，分别针对不同的游戏类型。刚体可以接受力和扭矩（通过脚本编写代码来实现），向刚体施加力/扭矩实际上会改变对象的变换组件位置和旋转。  
>    [2D 刚体官方文档](https://docs.unity3d.com/cn/2021.1/Manual/class-Rigidbody2D.html)

### 1.3 添加障碍物和碰撞

1. 添加小行星：在playground/Image/Asteroids路径下找到陨石，将其拖入Scene或Hierarchy中，然后修改这个gameobject名字为NPC_Asteroid_Stone_Type2_01，如果是第二块石头则NPC_Asteroid_Stone_Type2_02，以此类推，如图。将小行星大小，位置调整合适
2. 添加两个组件到小行星：Rigidbody2D 和 PolygonCollider2D，取消勾选“is Trigger”，否则collider效果无法作用在物理引擎上。
3. 添加 PolygonCollider2D 给飞船：必须都有该组件，才能正常碰撞，因为如果要模拟物体之间的碰撞的前提是他们都是刚体（Rigidbody）并且都有碰撞体积（Collider）。
4. 设置小行星重力为 0，否则一开始游戏物体都会向下掉。
5. 点击play，调整小行星参数：Angular Friction 为旋转阻力

   > 注意：
   > 让我们将 Mass 参数设置的大一些，比如 10，这样便会增加小行星的重量，在碰到飞船时才不至于快速飞出边界。

6. 碰撞后掉血：向小行星添加一个名为 ModifyHealthAttribute 的组件，如果之前创建没有创建Collider2D此时unity会弹出一个窗口Collider2D need，由于脚本与Collider2D有关联，提示我们需要为该物体创建一个碰撞体，由于我们陨石的形状是不规则的，因此我们选择Polygon（多边形），最后确认给小行星挂上Rigidbody2D 的组件，

   ![](../../imgs/unity_modhp.png)

7. 向飞船添加另外一个名为 HealthSystemAttribute 的组件，这样飞船能够检测到这种伤害。

   ![](../../imgs/unity_hpSysAttr.png)

8. 在Asset目录下新建一个Prefabs文件夹，通过将小行星拖拽到 project/Assets/Prefabs 中，将其设置为预制件(Prefab)；并进行复制（Windows 上的快捷键为 Ctrl+D，Mac 上的快捷键为 Command+D），或者直接将prefabs目录下的预制件直接拖拽到场景，从而在飞船周围创建一个小的小行星场。

> 相关概念：
>
> - Collider（碰撞体）：  
>   [官方文档](https://docs.unity3d.com/cn/2021.1/Manual/class-PolygonCollider2D.html)  
>   2D 多边形碰撞体，碰撞体的形状由线段组成的自由形状边缘定义，因此可对其进行高精度调整以适应精灵图形的形状。
> - Prefab (预制件):  
>    [预制件 官方文档](https://docs.unity3d.com/cn/2021.1/Manual/Prefabs.html)  
>   Unity 的预制件系统允许创建、配置和存储游戏对象及其所有组件、属性值和子游戏对象作为可重用资源。预制件资源充当模板，在此模板的基础之上可以在场景中创建新的预制件实例。  
>    说白了，就是“可重用资源模板”

### 1.4 添加用户界面

我们希望飞船在碰到陨石之后会扣除血量，但是这个数字要如何显示呢？显然这需要通过用户界面（user interface）给玩家回馈信息，这里就要用到unity的UGUI系统了

将 Playground/Prefabs/UserInterface 预制件从 /Prefabs 文件夹拖入到场景中。在你查看 Game 视图时，系统会自动显示一个 UI 弹出窗口，其中包含 Score 和 Health 属性。
  
  [UGUI 官方文档](https://docs.unity3d.com/Packages/com.unity.ugui@1.0/manual/index.html)

添加背景：找到../Images/Backgrounds 目录，可以看到我们这里有很多背景，这里我们选择BG_Space，将他拖拽到我们场景中。

但是放置背景之后我们发现屏幕并没有被背景铺满，而且陨石不见了，这里我们先要调整背景所在的图层，让他位于最下面才行。首先修改背景的Sprite Renderer的Draw Mode为“Tiled（已平铺）”，展开Visibility Options，将Sorting Layer 改为 Background，这样就将背景置于最下层了，最后修改Scale，让背景铺满整个屏幕。
### 1.5 添加游戏目标

游戏规则：这个游戏的目标是什么呢？假设我们希望飞船收集一些星星，但不能撞到小行星。一旦收集到所有星星，便赢得了比赛。——canvas显示You win!
但如果发生碰撞的次数太多，游戏便结束！——飞船消失，同时canvas显示Game Over！

1. 添加星星：  
   将这个星星（Star）从 ../Images/GameElements 文件夹拖入到场景中。
2. 添加分数相关组件：  
   为星星添加 Collectable 脚本，将星星变成一个可收集物品，可以看到该组件暴露的Public变量：Points Worth，收集到一个星星便会奖励玩家一个点数。出现Collider2D needed提示点击Polygon（多边形)，即创建一个多边形的Collider
3. 添加碰撞体，让星星能被飞船收集：  
   为星星添加 PolygonCollider2D 组件，并启用 Is Trigger 属性

4. 最后同样在Prefabs文件夹下为星星创建预制件，最后创建多个，在飞船周围创建一个小的小行星场。

> 相关概念：
>
> - 触发器：  
>    用于触发事件。如果启用 Is Trigger 属性，则该碰撞体将用于触发事件，并被物理引擎忽略。
>   触发后，可以让对象变为无形（比如：被吃掉的金币，被拾取的装备等），Unity 会检测到两个对象相互接触的情况，继而执行设定好的事件（执行事件函数）。

### 1.6 添加获胜条件

完善游戏，添加获胜条件

1. 将星星作为预制件，复制 5 个；将这些星星分布到周围，让一些星星很容易获得，但其他一些星星很难获得。通过这样的方式，我们也可以让我们的小游戏难度逐渐增大。
2. 选择 UI 游戏对象，然后在 UI 脚本上确保游戏类型为 Score，并且所需得分为 5。
3.修改player上挂在的Health System组件，调整Health控制难度，如果小于等于0只要碰到小行星就Game Over了。

> 注意：  
> 星星个数需要 大于等于 Score 值，否则永远无法胜利

## 2. 发挥自己想象力

发挥想象力，重复利用 PlayGround 中的各项资源，合理利用组件，创作属于自己的小游戏！

在学习如何制作游戏时，模仿 80 年代老游戏（例如小行星、打砖块、太空侵略者、青蛙等等……）的玩法一般来说是很不错的做法，因为这些游戏很简单。然后，随着你的水平不断提升，你可以添加越来越多的细节并优化互动方式。

如果你需要灵感，请打开 Examples 文件夹并启动其中一个游戏。检查游戏对象，看看我们是如何制作这些游戏对象的，然后尝试创建类似的游戏对象。

<br>
<hr>
<br>

配套视频教程：
[https://space.bilibili.com/43644141/channel/seriesdetail?sid=299912](https://space.bilibili.com/43644141/channel/seriesdetail?sid=299912)

文章也同时同步微信公众号，喜欢使用手机观看文章的可以关注

![](../../imgs/微信公众号二维码.jpg)
