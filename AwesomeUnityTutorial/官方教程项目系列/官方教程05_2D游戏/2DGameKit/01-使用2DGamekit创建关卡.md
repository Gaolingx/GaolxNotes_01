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

## 3. 按教程操作

<br>

<hr>
<br>

配套视频教程：
[https://space.bilibili.com/43644141/channel/seriesdetail?sid=299912](https://space.bilibili.com/43644141/channel/seriesdetail?sid=299912)

文章也同时同步微信公众号，喜欢使用手机观看文章的可以关注

![](../../imgs/微信公众号二维码.jpg)
