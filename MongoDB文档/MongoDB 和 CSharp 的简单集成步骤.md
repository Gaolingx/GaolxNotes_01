在 C# 中使用 MongoDB，可以通过官方提供的 **MongoDB .NET Driver** 来操作数据库。

## 驱动安装
### NuGet

在项目中通过 NuGet 安装 MongoDB 的官方 .NET 驱动程序:

```bash
Install-Package MongoDB.Driver
```

或使用 .NET CLI：

```bash
dotnet add package MongoDB.Driver
```

### Github仓库

你还可以通过仓库方式安装源代码：

```bash
https://github.com/mongodb/mongo-csharp-driver
```

该方法适合魔改的用户使用。

### 选择驱动安装方式

个人更加偏向NuGet方式安装，因为这种方式升级相对比较简单，相反源代码升级起来就要麻烦一些。如果没有一些特殊个人定制的取消，建议使用NuGet就可以。

## 代码中使用MongoDB

### 创建 MongoDB 数据库和集合

在 MongoDB 中，数据库和集合会在插入数据时自动创建，因此不需要手动创建。
### 连接到 MongoDB

使用 `MongoClient` 连接到 MongoDB。

```csharp
using MongoDB.Driver;

public class MongoDBConnection
{
    private readonly IMongoDatabase _database;

    public MongoDBConnection(string databaseName)
    {
        // MongoDB 默认连接字符串
        var client = new MongoClient("mongodb://localhost:27017");
        _database = client.GetDatabase(databaseName);
    }

    public IMongoCollection<T> GetCollection<T>(string collectionName)
    {
        return _database.GetCollection<T>(collectionName);
    }
}
```
### 创建模型类

MongoDB 的集合中的文档通常是 JSON 格式，可以用 C# 类来映射。

```csharp
using MongoDB.Bson;
using MongoDB.Bson.Serialization.Attributes;

public class User
{
    [BsonId] // 指定主键
    [BsonRepresentation(BsonType.ObjectId)] // 使用 ObjectId 类型
    public string Id { get; set; }

    [BsonElement("name")] // 指定字段名
    public string Name { get; set; }

    [BsonElement("age")]
    public int Age { get; set; }
}
```
### 插入数据

使用 `InsertOne` 或 `InsertMany` 方法插入数据。

```csharp
public void InsertUser()
{
    var dbConnection = new MongoDBConnection("TestDatabase");
    var userCollection = dbConnection.GetCollection<User>("Users");

    var newUser = new User
    {
        Name = "Alice",
        Age = 25
    };

    userCollection.InsertOne(newUser);
    Console.WriteLine("User inserted successfully!");
}
```
### 查询数据

使用 `Find` 方法查询集合中的数据。

```csharp
public void QueryUsers()
{
    var dbConnection = new MongoDBConnection("TestDatabase");
    var userCollection = dbConnection.GetCollection<User>("Users");

    // 查询所有用户
    var users = userCollection.Find(user => true).ToList();

    foreach (var user in users)
    {
        Console.WriteLine($"ID: {user.Id}, Name: {user.Name}, Age: {user.Age}");
    }

    // 条件查询（如查询年龄大于 20 的用户）
    var filteredUsers = userCollection.Find(user => user.Age > 20).ToList();

    foreach (var user in filteredUsers)
    {
        Console.WriteLine($"Filtered -> ID: {user.Id}, Name: {user.Name}, Age: {user.Age}");
    }
}
```
### 更新数据

使用 `ReplaceOne` 或 `UpdateOne` 方法更新数据。

```csharp
public void UpdateUser(string id)
{
    var dbConnection = new MongoDBConnection("TestDatabase");
    var userCollection = dbConnection.GetCollection<User>("Users");

    var filter = Builders<User>.Filter.Eq(user => user.Id, id);
    var update = Builders<User>.Update.Set(user => user.Age, 30);

    var result = userCollection.UpdateOne(filter, update);
    Console.WriteLine($"{result.ModifiedCount} document(s) updated.");
}
```
### 删除数据

使用 `DeleteOne` 或 `DeleteMany` 方法删除数据。

```csharp
public void DeleteUser(string id)
{
    var dbConnection = new MongoDBConnection("TestDatabase");
    var userCollection = dbConnection.GetCollection<User>("Users");

    var filter = Builders<User>.Filter.Eq(user => user.Id, id);
    var result = userCollection.DeleteOne(filter);

    Console.WriteLine($"{result.DeletedCount} document(s) deleted.");
}
```
### 完整的代码示例

以下是一个完整的控制台应用程序示例：

```csharp
using System;
using MongoDB.Bson;
using MongoDB.Driver;

namespace MongoDBExample
{
    public class User
    {
        [BsonId]
        [BsonRepresentation(BsonType.ObjectId)]
        public string Id { get; set; }

        [BsonElement("name")]
        public string Name { get; set; }

        [BsonElement("age")]
        public int Age { get; set; }
    }

    public class MongoDBConnection
    {
        private readonly IMongoDatabase _database;

        public MongoDBConnection(string databaseName)
        {
            var client = new MongoClient("mongodb://localhost:27017");
            _database = client.GetDatabase(databaseName);
        }

        public IMongoCollection<T> GetCollection<T>(string collectionName)
        {
            return _database.GetCollection<T>(collectionName);
        }
    }

    class Program
    {
        static void Main(string[] args)
        {
            var dbConnection = new MongoDBConnection("TestDatabase");
            var userCollection = dbConnection.GetCollection<User>("Users");

            // 插入数据
            var newUser = new User { Name = "Alice", Age = 25 };
            userCollection.InsertOne(newUser);
            Console.WriteLine("User inserted.");

            // 查询数据
            var users = userCollection.Find(user => true).ToList();
            Console.WriteLine("All users:");
            foreach (var user in users)
            {
                Console.WriteLine($"ID: {user.Id}, Name: {user.Name}, Age: {user.Age}");
            }

            // 更新数据
            var filter = Builders<User>.Filter.Eq(user => user.Name, "Alice");
            var update = Builders<User>.Update.Set(user => user.Age, 30);
            var result = userCollection.UpdateOne(filter, update);
            Console.WriteLine($"{result.ModifiedCount} document(s) updated.");

            // 删除数据
            var deleteFilter = Builders<User>.Filter.Eq(user => user.Name, "Alice");
            var deleteResult = userCollection.DeleteOne(deleteFilter);
            Console.WriteLine($"{deleteResult.DeletedCount} document(s) deleted.");
        }
    }
}
```

---
## 使用建议

### 索引优化

为查询频繁的字段创建索引，提高查询性能。

```csharp
userCollection.Indexes.CreateOne(new CreateIndexModel<User>(
    Builders<User>.IndexKeys.Ascending(user => user.Name)
));
```
### 连接池管理

`MongoClient` 是线程安全的，建议在整个应用程序中复用一个实例，而不是每次都创建新实例。
### 异常处理

捕获 MongoDB 的异常（如 `MongoException`），记录日志并妥善处理。