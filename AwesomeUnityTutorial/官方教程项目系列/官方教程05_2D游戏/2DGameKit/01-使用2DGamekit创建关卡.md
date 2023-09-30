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
  
<br>

<hr>
<br>

配套视频教程：
[https://space.bilibili.com/43644141/channel/seriesdetail?sid=299912](https://space.bilibili.com/43644141/channel/seriesdetail?sid=299912)

文章也同时同步微信公众号，喜欢使用手机观看文章的可以关注

![](../../imgs/微信公众号二维码.jpg)
