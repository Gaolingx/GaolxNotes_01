# 使用 2D Game Kit 创建关卡

[2D 游戏套件演练-官方教程](https://learn.unity.com/tutorial/2d-you-xi-tao-jian-yan-lian?uv=2020.3&projectId=5facff3cedbc2a001f5338ab#)

## 1. 导入
  
在学习我们这门课程之前当然需要先准备好资源，一般在Asset Store里面搜索《2D Game Kit》按照之前的方式添加到我的资源再到unity用Package Manager下载并导入Package所有文件到一个空项目即可。
  
如果网络不好的也可以先把unitypackage下载到本地再手动导入
> 本地 unitypackage 文件使用：  
> 新建项目 --> 打开项目 --> 将 unitypackage 文件拖入已打开 unity 的界面的 Project 窗口中 --> 选择需要导入的资源，import 导入
> 
> - [AssetStore 地址](https://assetstore.unity.com/packages/templates/tutorials/2d-game-kit-107098?_ga=2.162437502.331241089.1633678521-522971275.1624332126)
> - [Baidu 云盘](https://pan.baidu.com/s/12GoDAXrZd_PCiYDbZV_gpw) 提取码：6nha
> - [迅雷云盘](https://pan.xunlei.com/s/VMl_AAmxjqPxnJQWlfnxcQJpA1) 提取码：9i4x

## 2. 改初始 Bug
  
unity中提示错误时候必须引起足够的重视，它会影响正常开发流程，使得我们无法正常运行，所有掌握排除错误的技能也是客户端开发很重要的技能
  
通常情况下，unity控制台一段错误提示是由脚本物理路径+出错的行列+c#的错误代码+错误详细描述。你可以将关键字or一整串错误代码进行搜索帮助你解决问题
  
2.1 RuleTile 类冲突

在项目中，自定义了一个 RuleTile 类，和unity包中的同名类（Unity.2D.Tilemap.Extras）冲突，说白了就是c#不允许出现两个相同的类处于同一命名空间，会导致导入后第一时间提示出错

改错方法很简单：重命名项目中的 RuleTile 类、代码加命名空间（比较麻烦）

1. Assembly-CSharp\Assets\2DGameKit\Scripts\Utility\RuleTile.cs 重命名为 RuleTile2dgk.cs ，并且类名也要更改。推荐使用 visual studio 中“重命名”操作，vs中操作方法是选中要修改的类名，右键点击“重命名”或者按两次ctrl+r，弹出重命名工具，输入新的名字，按enter应用更改；
2. Assembly-CSharp-Editor\Assets\2DGamekit\Utilities\Editor\RuleTileEditor.cs 中所有原有的 RuleTile ，替换为 RuleTile2dgk ，推荐使用 visual studio 中“替换查找操作，vs中操作方法是按ctrl+f，展开，查找填上RuleTile，替换填上RuleTile2dgk，选中“匹配大小写”和“全字匹配”，按alt+a全部替换，最后别忘了保存

完成以上操作后，可以看到unity控制台已经不报错了，至此我们旧掌握了如何排查脚本错误以及通过vs完成重命名和批量替换的操作。

3.1 Cinemachine命名空间缺失
  
由于我们新建的项目默认不会添加Cinemachine组件，需要在Package Manager中手动添加对应的Package。

[RuleTile 使用教程](https://learn.unity.com/tutorial/using-rule-tiles#)

[2D TileMap Extras](https://docs.unity3d.com/Packages/com.unity.2d.tilemap.extras@2.2/manual/index.html)
  
【百日挑战38】unity教程之2D游戏开发初步（三）
  
前言：大家好，今天是中秋国庆双节8天长假的第二天，这里小高祝大家双节快乐，在上期教程中，我们开始初步开始使用官方项目《2DGameKit》逐渐熟悉2D平台游戏开发流程，学习了如何查错以及使用vs进行快速操作，今天我们通过项目预设好的半成品关卡来设计2D游戏关卡，我们先学习创建场景以及如何使用瓦片调色板编辑地图。
  
## 3. 按教程操作
  
3.1 创建场景：  
   由于直接创建新的空场景话操作太过繁琐，在顶部菜单栏中，选择 Kit Tools > Create new Scene。在弹出的NewSceneCreator面板中输入新场景的名字 MainScene ，点击Create，即可在 Asset 里创建新的场景，建议在assets目录单独建一个Scene目录便于统一管理场景文件。

   > 注意：  
   > Kit Tools 菜单并非是 unity 中的默认菜单，而是通过代码，新增如 Unity 编辑器的自定义功能。
   > unity 支持定制功能，非常灵活，开发者可以根据自己的需求，打造属于自己的 unity ，从而简化开发过程。

可以看到，脚本默认为我们创建了一个已经包含部分游戏对象的场景，其中包含了背景、玩家、一个瓦片地图等等，类似一个模板，降低了前期繁琐的工作。
  
3.2 了解场景游戏对象  
通过Hierarchy可以清楚的看到我们场景中各个游戏对象和他们的作用，可以看到，这里使用了空游戏对象来进行分类，将同组的游戏对象放在一起，适用于不希望将游戏对象放入父子关系但又希望层级视图中对各个游戏对象进行分组，这是一个很不错的好习惯。
  
通过阅读Hierarchy的分组，可以将游戏对象大致分为几个类型：
· Gameplay：游戏逻辑，玩法相关的逻辑
· GameUtilities：构成游戏基本组件（摄像机，特效管理器、场景管理器、转场管理器、物理Handle等等）
· UI：游戏用户界面，显示血条、加载、GameOver等画面
· Characters：玩家对象
· LevelArt：场景中的瓦片地图、背景等等美术风格有关的东西
  
3.3 构建平台
简单了解完这个关卡的构成，接下来我们学习操作trie map为角色设计前进的路线。Tilemap是一个比较复杂的模块，我这里只简单提下
  
1、选中TilemapGrid，点击“打开瓦片调色板”，里面有两种Palette（调色板）供我们选择 ，这里选择 TilesetGameKit，选中一个网格，功能菜单点击brush，菜单其他功能参考下图
  
2、在scene中，按鼠标左键拖动即可编辑，鼠标中键拖动鼠标中键可平移视图，选中erase可以擦除多余的瓦片，拐弯抹角的地方unity会自动帮我们处理
  
3、构建完瓦片地图后，就可以在地图内操作角色进行移动了，碰撞体和刚体脚本已经自动为我们创建。
  
创建完了承载玩家和npc的静态平台，再让我们继续创建动态平台，以下用创建一个移动平台（MovingPlatform）为例
  
1、简单预览下这个Prefab，可以看到它包括了两个脚本和三个物理有关的组件，MovingPlatform脚本控制移动速度、方位、点位、循环模式等等属性
  
2、使用方法也很简单，将这个预制件拖进scene即可，最后不要忘了打开scene视图的Gizmos，否则坐标轴、移动范围等图标都会不显示，影响开发
  
3、添加node，设置node位置，设置速度，MovingPlatformType设置为LOOP，保存运行，可以看到平台就会在node所在位置之间来回移动了。至此，我们就完成了2D游戏中重要元素——平台的创建
  
【百日挑战39】unity教程之2D游戏开发初步（四）
  
前言：大家好，今天是中秋国庆双节8天长假的第三天，这里小高祝大家双节快乐，在上期教程中，我们开始初步开始使用官方项目《2DGameKit》创建场景以及如何使用瓦片调色板编辑地图，搭建静态平台和移动平台，今天我们继续通过项目预设好的半成品关卡来设计2D游戏关卡，学习创建更多交互性的对象，比如创建一个可以开门的开关等等机关。
  
3.4 设计交互方案  
我们希望玩家踩在一个压力板上，对应的机关会进行一系列操作（如对象的移动），通过一个游戏对象控制一个或多个其他的游戏对象。
  
3.4.1 设计与玩家交互物品（触发器）  
1、定位 Assets\2DGamekit\Prefabs\Interactables 目录，找到PressurePad（压力板）的prefab，双击打开进入预制件编辑模式
  
2、这里只介绍几个主要的组件，注意其中的 PlatformCatcher 脚本和 PressurePad 脚本，待会将调整组件内的值
  
3、将预制件 PressurePad 拖入scene中，调整好位置，在Box Collider 2D组件中点击“编辑碰撞器”即可调整这个机关生效的范围以及位置，注意不要让机关的collider处于平台的collider之下，否则角色可能无法触发机关，可以调整 PressurePad 位置也可以调整 PressurePad 挂载的 collider 位置。
  
3.4.2 设计触发的效果（效果器）
我们希望Player触发压力板之后触发游戏对象操作——开门。
  
1、定位 Assets\2DGamekit\Prefabs\Interactables 目录，找到Door（门）的prefab，将它拖放到scene位置中，调整位置
  
2、由于我们的角色是可以跳跃的，我们不希望玩家能跳过这个door，我们需要在特定时候限制玩家移动范围，你可以用空的游戏对象挂载Box Collider 2D组件做一道空气墙，也可以通过瓦片调色板加在门上挡住玩家，都是可以的
  
3.4.3 关联触发器与效果器  
1、回到 PressurePad 的PressurePad 组件，要实现开门的效果，我们需要在 Boxes 里的 OnPressed() 的UnityEvent 里添加新的对象，即将踩踏板与Door开门的这个事件关联起来（为该事件添加监听对象），当玩家触发了 PressurePad 就会通知 Door 触发开门的效果，这里用到的unity事件函数以后再详细讲
  
2、点击加号添加新的事件，将Hierarchy中的Door拖放到新建事件的监听对象中，Function 选择 Animator->Play (string)，下面是对应的参数，这里填写的是播放动画状态机对应的名称（区分大小写，不能填错）
  
3、保存运行，看效果，可以看到当角色与PressurePad 挂载的 collider接触到时候，门就会自动打开了（scene中可以看到门隐藏在了平台的下面），至此我们就完成了交互机关的设计。

注：事件函数：Unity 中的脚本与传统的程序概念不同。在传统程序中，代码在循环中连续运行，直到完成任务。相反，Unity 通过调用在脚本中声明的某些函数来间歇地将控制权交给脚本。函数执行完毕后，控制权将交回 Unity。这些函数由 Unity 激活以响应游戏中发生的事件，因此称为事件函数。常见的事件函数有Start方法、Update方法等。
  
【百日挑战40】unity教程之2D游戏开发初步（五）
  
前言：大家好，今天是中秋国庆双节8天长假的第四天，这里小高祝大家双节快乐，在上期教程中，我们开始初步开始使用官方项目《2DGameKit》创建场景以及如何使用瓦片调色板编辑地图，搭建静态平台和移动平台，创建交互机关，今天我们继续通过项目预设好的半成品关卡来设计2D游戏关卡，学习如何加入非玩家角色（npc）以及调整各项参数增强游戏性。
  
3.5 设计npc
npc可以推动剧情的发展，丰富游戏的内涵与玩法，常见的npc通常分为两类，一种是可攻击的npc（例如怪物等），一种是可对话的npc（例如剧情交互角色），这里演示下如何创建可攻击的npc。
  
1、定位 Assets\2DGamekit\Prefabs\Enemies 目录，其中有三个预制件，Spitter 和 Chomper 分别是远程和近战的怪物，EnemySpawner是怪物的生成器，选中其中一个Enemies的预制件，CharacterController2D组件负责玩家的输入，Damageable组件负责血量和伤害的管理，以及受到伤害触发的效果（能被攻击和消灭的npc需要这个组件），我们主要关心的是 EnemyBehaviour 组件，它主要负责npc核心的表现层控制
  
2、简单介绍下EnemyBehaviour 组件，speed控制移动速度，gravity控制重力，projectilePrefab控制要投射的物品（预制件），Scanning settings主要是设置怪物可见攻击范围，玩家超过这个可视范围就不会攻击，viewDirection控制视角方向，viewFov控制视角大小，viewDistance控制可视范围，timeBeforeTargetLost 控制在目标被视为从视线中消失之前，目标不在视锥中的时间（以秒为单位）。Melee Attack Data控制近战攻击相关参数，meleeRange 控制攻击范围，meleeDamager指怪物自身技能造成的伤害，contactDamager 指玩家接触到怪物造成的伤害，attackForce 控制攻击的力度，后面就是绑定不同状态下触发声音相关的组件了
  
3、将怪物们拖拽到Scene中以创建他们，绿色部分表示怪物可视范围，红色表示攻击范围，试着通过修改EnemyBehaviour 组件中viewDirection,viewFov,viewDistance控制视野和攻击范围。
  
3.6 设计npc的生成器
我们在地图中设置一个固定刷怪点，totalEnemiesToBeSpawned 表示这个刷怪点一共能刷多少只怪，concurrentEnemiesToBeSpawned表示一次刷多少只怪，spawnArea是生成范围，spawnDelay是生成的冷却时间，removalDelay是怪物被移除后重新出现的延迟。
  
运行，可以看到，当我们打死上面的怪后，过了removalDelay设定的时间怪重新生成了。
  
从上不难看出，设置可交互的游戏对象对于gameplay有很重要的影响，可以提高游戏的可玩性，有时候，我们希望除了直接攻击怪物，我们也可以通过触发机关来攻击怪物，除了之前我们提到的压力板与门的关联，我们也可以将这样的关联设置在怪物上面。例如，我们创建一个能推动的箱子，用箱子从高处掉落砸这个怪物。下期教程我们将探索如何让玩家推箱子间接消灭怪物的玩法。
  
【百日挑战41】unity教程之2D游戏开发初步（六）  
前言：大家好，今天是中秋国庆双节8天长假的第五天，这里小高祝大家双节快乐，在上期教程中，我们开始初步开始使用官方项目《2DGameKit》创建场景以及如何使用瓦片调色板编辑地图，搭建静态平台和移动平台，创建交互机关，加入非玩家角色（npc），今天我们继续通过项目预设好的半成品关卡来设计2D游戏关卡，学习如何设计玩家与游戏对象之间的互动——设计让玩家推箱子间接消灭怪物的玩法。
  
3.7 设计一个可以推动的箱子
  
1、定位到 Assets\2DGamekit\Prefabs 目录，其中是一些可互动的预制件，找到 PushableBox 预制件，它是一个可以被推动的箱子，玩家从哪边推就往哪里走，而且受到重力影响，双击进入预制件编辑模式，观察这个预制件，Pushable 组件负责处理力与碰撞以及声音相关的逻辑，比较特别的是它有三个盒形碰撞体，由于要求玩家不能穿过而且要能落在平台上，所以一个处理与地形和其他物品的碰撞，还有两个响应玩家推箱子这个操作的。
  
2、将这个预制件拖入 scene，设计一个高台和一个怪物，把箱子放在高台上，调整位置让玩家可以推下去，同时保证能砸中怪物。
  
3、摆好箱子的位置后，我们希望箱子接触到怪物能造成伤害，我们需要为这个箱子添加一个 Damager 的组件用于处理伤害有关的逻辑，表示这个伤害发出者，以及处理伤害逻辑，调用各种事件函数，这里要先设置下 hittableLayers 的 LayerMask 过滤伤害发生的层级，说白了就是让伤害发生在哪些层级，这里只选中 Enemy，最后调整伤害范围盒的偏移与大小，与盒子大致重合即可。
  
4、保存，测试游戏，让玩家尝试推动这个箱子，检查箱子是否可以被推动和正常掉落，以及当箱子掉落在 Enemy 上时敌人是否被消灭
  
最后运行的效果如图，可以看到敌人被箱子砸到之后被设置为不可见了
  
至此，我们就完成了玩家与多个游戏对象直接和间接的互动，类似这样的玩法可以参考《超级马里奥》，里面就大量运用到了玩家与游戏对象之间的互动，比如玩家可以吃蘑菇升级，碰到怪会减少血量，玩家通过跳跃触发某些隐藏机关等等，可以说，现代的游戏中，玩家与游戏对象（npc，敌人，其他交互物等）的交互很大程度上丰富了游戏的核心玩法甚至成为推动游戏剧情的关键一环，希望通过两期的学习，大家可以尝试着创建更丰富的地图和更多的交互物丰富你的游戏玩法。

<br>

<hr>
<br>

配套视频教程：
[https://space.bilibili.com/43644141/channel/seriesdetail?sid=299912](https://space.bilibili.com/43644141/channel/seriesdetail?sid=299912)

文章也同时同步微信公众号，喜欢使用手机观看文章的可以关注

![](../../imgs/微信公众号二维码.jpg)
