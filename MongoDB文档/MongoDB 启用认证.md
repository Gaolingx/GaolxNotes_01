MongoDB 默认情况下是不启用身份验证的，您需要修改配置文件或参数来启用它。

## 1、什么是Authentication（认证）

Authentication 是验证用户的身份，授予用户对资源和操作的访问权限

简而言之，认证是告诉我“你是谁”，授权是“你可以做什么”

访问控制很简单，只有三步：

1. 启用访问控制
2. 创建用户
3. 认证用户 

## 2、启动不带Authentication的服务

默认情况下，数据库是没有任何用户账号的，也就是说，在未启用访问控制时，任何人都可以直接访问数据库而无需验证身份。但是，一旦启动了访问控制功能，就必须通过指定的账号和密码进行登录验证，才能访问数据库。这种机制的目的是为了增强数据的安全性，防止未经授权的用户获取敏感数据或对数据库进行操作。

所以应该先启用访问控制的服务，在admin数据库中创建用户管理员账号并且赋予权限，然后用用户管理员登录（或者叫身份验证）成功以后才可以再创建其它用户。当然也可以直接使用这个账号进行数据库操作。

- 命令行参数方式启动

```css
mongod --bind_ip 127.0.0.1 --port 27017 --dbpath c:\data
```

- 配置文件方式启动

	首先在mongod.conf或mongod.cfg文件里找到以下内容并注释掉，如本来就没有那就不用修改

```yaml
#security:
    #authorization: enabled
```

	然后执行命令启动服务

```css
mongod --config c:\mongod.cfg
```

## 3、数据库中添加账户

  1. 使用 MongoDB 提供的命令行工具连接到数据库中
  2. 切换到admin数据库
  3. 添加一个myUserAdmin用户，并赋予其“userAdminAnyDatabase”和“readWriteAnyDatabase”这两个角色

```css
use admin
db.getUsers()  // 可以先用这个命令查询下当前数据库中的所有账号
db.createUser(
  {
    user: "myUserAdmin",
    pwd: passwordPrompt(), // or cleartext password
    roles: [
    	{ role: "userAdminAnyDatabase", db: "admin" },
      	{ role: "readWriteAnyDatabase", db: "admin" }
    ]
  }
)
```

注意：passwordPrompt()方法提示你输入密码。你也可以直接将密码指定为字符串。建议使用passwordPrompt()方法，以避免在屏幕上看到密码，并可能将密码泄露到shell历史记录中。

- `userAdminAnyDatabase` 角色授予用户管理所有数据库的权限。
- `readWriteAnyDatabase` 角色允许用户对所有数据库执行读写操作。

## 4、重新启动MongoDB服务

请务必避免直接使用 **CTRL+C** 或直接点击“停止服务”按钮来关闭 MongoDB 服务。这种强制关闭操作可能会导致数据未正确写入磁盘，进而损坏 MongoDB 的数据库文件。一旦发生文件损坏，下次启动时可能会出现无法预料的错误，甚至导致数据丢失或系统崩溃。

正确关闭 MongoDB 的方法是通过以下命令优雅地停止服务，以确保所有正在进行的操作完成并安全地将数据写入磁盘：

- **使用 MongoDB 提供的命令行工具**：

``` css
// 该命令会优雅地关闭服务，确保所有缓冲区的内容写入磁盘。
db.adminCommand({ shutdown: 1 })
```

- **通过系统服务管理工具关闭**（适用于服务已注册的情况）：

```bash
// 在 Linux 系统中，可以使用以下命令停止并重新启动服务
sudo systemctl restart mongod
```

```bash
// 在 Windows 系统中，可以通过服务管理器或执行以下命令停止
net stop MongoDB
// 执行以下命令启动
net start MongoDB
```

## 5、启动Authentication的服务

- 命令行参数方式启动

```css
mongod --auth --bind_ip 127.0.0.1 --port 27017 --dbpath c:\data
```

- 配置文件方式启动

	首先在mongod.conf或mongod.cfg文件里找到以下内容并删掉注释，如没有就把下面内容粘贴到文件中

```yaml
security:
    authorization: enabled
```

	然后执行命令启动服务

```css
mongod --config c:\mongo.conf
```

## 6、连接字符串特殊符号

在连接 MongoDB 时，如果用户名或密码中包含特殊字符（如 `@`、`/`、`:`、`?` 等），你需要对这些特殊字符进行 URL 编码。URL 编码是将字符转换为可以在 URL 中安全传输的格式。这是因为某些字符在 URL 中具有特定的含义，比如 `@` 用于分隔用户名和主机，`:` 用于分隔主机和端口。

### URL 编码的基本规则

URL 编码的格式是将特殊字符替换为 `%` 后跟其 ASCII 十六进制值。例如：

- 空格编码为 `%20`
- `@` 编码为 `%40`
- `/` 编码为 `%2F`
- `:` 编码为 `%3A`
- `?` 编码为 `%3F`

### 示例

假设您的用户名是 `user@example.com`，密码是 `pa/ss:word?`，那么在连接字符串中，您需要这样处理：

- 用户名 `user@example.com` 中的 `@` 需要编码为 `%40`
- 密码 `pa/ss:word?` 中的 `/` 需要编码为 `%2F`，`:` 需要编码为 `%3A`，`?` 需要编码为 `%3F`

因此，编码后的用户名和密码分别为：

- 编码后的用户名：`user%40example.com`
- 编码后的密码：`pa%2Fss%3Aword%3F`

### 最终连接字符串

将编码后的用户名和密码放入连接字符串中，结果如下：

```bash
mongodb://user%40example.com:pa%2Fss%3Aword%3F@<主机>:<端口>/<数据库>
```

### 如何进行 URL 编码

1. 网上搜索URL编码有很多在线工具可以编码
2. 使用编程语言进行编码
3. 使用如DevToys这样的工具进行编码

## 7、数据库权限角色

每个MongoDB的内置角色都定义了角色数据库中所有非系统集合的数据库级别的访问权限以及所有系统集合的集合级别的访问权限。

MongoDB为每个数据库提供内置的数据库用户和数据库管理角色。MongoDB仅在admin数据库上提供所有其他内置角色。

|角色类型|角色名称|角色描述|
|---|---|---|
|Database User Roles|read|提供在所有非系统集合和system.js集合中读取数据的能力。|
|readWrite|提供“read”角色的所有权限，外加在所有非系统集合和system.js集合中修改数据的能力。|
|Database Administration Roles|dbAdmin|提供执行管理任务的能力，例如与数据库相关的任务、索引和收集统计信息。该角色不授予用户和角色管理权限。|
|dbOwner|数据库所有者可以对数据库执行任何管理操作。该角色结合了readWrite、dbAdmin和userAdmin三个角色的权限。|
|userAdmin|提供在当前数据库上创建和修改角色和用户的能力。|
|Cluster Administration Roles|clusterAdmin|提供最大的集群管理访问。此角色结合了 clusterManager、clusterMonitor 和 hostManager 角色授予的权限。此外，该角色还提供 dropDatabase 操作的权限。|
|clusterManager|提供对集群的管理和监控操作的权限|
|clusterMonitor|提供对监控工具的只读访问权限|
|hostManager|提供监控和管理服务器的能力|
|Backup and Restoration Roles|backup|提供备份数据所需的最低权限|
|restore|提供从备份中恢复数据所需的权限|
|All-Database Roles|readAnyDatabase|提供和“read”角色相似的在所有数据库上的只读权限，local和config数据库除外|
|readWriteAnyDatabase|提供和“readWrite”角色相似的在所有数据库上的读写权限，local和config数据库除外|
|userAdminAnyDatabase|提供和“userAdmin”角色相似的在所有数据库上访问用户管理操作的权限，local和config数据库除外|
|dbAdminAnyDatabase|提供和“dbAdmin”角色相似的在所有数据库上管理操作的权限，local和config数据库除外|
|Superuser Roles|root|超级用户，提供以下角色的所有权限：<br><br>- readWriteAnyDatabase<br>- dbAdminAnyDatabase<br>- userAdminAnyDatabase<br>- clusterAdmin<br>- restore<br>- backup|
|Internal Role|__system|MongoDB 将此角色分配给代表集群成员的用户对象，例如副本集成员和 mongos 实例。|



