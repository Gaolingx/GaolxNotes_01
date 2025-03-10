前面我们安装的是 **MongoDB 的核心服务功能**，也就是服务器端数据库部分。需要注意的是，这个部分 **不包含任何可视化的管理工具**。

## 管理工具选择

在管理工具方面，**选择非常多**，可以根据个人需求和使用习惯进行选择。

### 官方提供的两种管理工具

1. **MongoDB Shell（mongosh）**： - 这是一个 **命令行工具**，主要用于通过命令行操作 MongoDB 数据库。 - 它适合那些熟悉命令行或者希望通过脚本自动化管理数据库的用户。

2. **MongoDB Compass**： - 这是一个带有图形用户界面（GUI）的工具。 - **Compass 提供了直观的操作界面**，方便用户通过点击和可视化的方式管理和查询数据库。 - 对于不熟悉命令行的用户，Compass 是一个不错的选择。

### 第三方工具

除了官方工具之外，还有很多优秀的第三方管理工具，比如 **Robo 3T**、**Studio 3T** 等。 由于工具种类繁多，具体使用哪种工具可以根据个人喜好选择，这里就不一一赘述了。

为了避免大家使用不同的工具，后面这里统一使用[dbgate](https://dbgate.org/download/)作为数据库工具。

---

## 使用命令行工具（MongoDB Shell）

### 下载

首先，前往 [MongoDB 官方网站](https://www.mongodb.com/try/download/shell) 下载 **MongoDB Shell**。这里选择下载 **Windows 版本**。

如果你正在观看视频教程，建议 **暂停视频**，先完成工具的下载和安装，再继续操作。

### 启动

下载完成后，你会得到一个可执行文件。双击启动 `mongosh` 可执行文件。

### 连接数据库

启动后，Shell 会提示你输入一个 **MongoDB 连接字符串**

默认的格式看起来很像浏览器中的网址。例如：

```plaintext
mongodb://localhost:27017
```

其中：

- **`localhost`** 表示连接本地的 MongoDB 实例。
- **`27017`** 是 MongoDB 的默认端口号。

输入正确的连接字符串后，按回车即可连接到数据库。接下来就可以通过命令行对数据库进行操作了。

### 查看数据服务下的所有数据库

```bash
show dbs
```

### 切换操作的数据库

```bash
use fantasy
```

### **插入数据**

```javascript
use myDatabase // 切换到目标数据库
db.users.insertOne({
    username: "张三",
    age: 25,
    gender: "男"
});
```


### 显示当前数据库中的集合（表）

```bash
> show collections
# 或
> show tables
```
