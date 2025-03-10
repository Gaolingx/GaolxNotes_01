在 C# 中使用 **BSON** 格式与 MongoDB 交互是完全支持的。MongoDB 的文档本质上是 BSON 格式，**MongoDB.Driver** 提供了对 BSON 的直接操作能力。以下是如何通过 BSON 格式插入和查询数据的详细指南。
## BSON 格式的基本操作

在 MongoDB 中，BSON 是一种二进制 JSON 格式。C# 的 **MongoDB.Bson** 命名空间提供了相关类（如 `BsonDocument` 和 `BsonValue`），可以用来直接操作 BSON 数据。
### 插入 BSON 格式的数据

使用 `BsonDocument` 构建 BSON 数据并插入到 MongoDB 中。

```csharp
using MongoDB.Bson;
using MongoDB.Driver;

public class BsonExample
{
    public void InsertUsingBson()
    {
        // 创建 MongoDB 客户端
        var client = new MongoClient("mongodb://localhost:27017");
        var database = client.GetDatabase("TestDatabase");
        var collection = database.GetCollection<BsonDocument>("Users");

        // 构造 BSON 数据
        var bsonDoc = new BsonDocument
        {
            { "name", "John Doe" },
            { "age", 29 },
            { "email", "john.doe@example.com" },
            { "isActive", true },
            { "address", new BsonDocument
                {
                    { "street", "123 Main St" },
                    { "city", "New York" },
                    { "zipCode", 10001 }
                }
            },
            { "hobbies", new BsonArray { "reading", "traveling", "coding" } }
        };

        // 插入数据
        collection.InsertOne(bsonDoc);
        Console.WriteLine("BSON document inserted successfully!");
    }
}
```

 **说明**：

- **`BsonDocument`**：一个动态文档，允许你构建嵌套结构。
- **`BsonArray`**：用于表示数组。
- 数据类型自动支持 `string`、`int`、`bool` 等等。

---
### 查询 BSON 格式的数据

使用 `BsonDocument` 查询数据。

```csharp
public void QueryUsingBson()
{
    // 创建 MongoDB 客户端
    var client = new MongoClient("mongodb://localhost:27017");
    var database = client.GetDatabase("TestDatabase");
    var collection = database.GetCollection<BsonDocument>("Users");

    // 查询所有文档
    var allDocuments = collection.Find(new BsonDocument()).ToList();

    foreach (var doc in allDocuments)
    {
        Console.WriteLine(doc.ToJson());
    }

    // 条件查询：查找年龄大于 25 的用户
    var filter = new BsonDocument("age", new BsonDocument("$gt", 25));
    var filteredDocuments = collection.Find(filter).ToList();

    Console.WriteLine("Filtered Documents:");
    foreach (var doc in filteredDocuments)
    {
        Console.WriteLine(doc.ToJson());
    }
}
```

**说明**：

- 空的 `BsonDocument()` 表示查询所有数据。
- **条件查询**：
    - 使用嵌套 `BsonDocument` 表达条件，例如 `{ "age": { "$gt": 25 } }`。
    - `$gt` 表示大于，类似的还有 `$lt`（小于）、`$eq`（等于）等。

---
### 使用 BSON 更新数据

通过 `UpdateOne` 方法更新数据，更新条件和更新内容均使用 BSON 格式。

```csharp
public void UpdateUsingBson()
{
    var client = new MongoClient("mongodb://localhost:27017");
    var database = client.GetDatabase("TestDatabase");
    var collection = database.GetCollection<BsonDocument>("Users");

    // 更新条件：查找名字为 "John Doe" 的用户
    var filter = new BsonDocument("name", "John Doe");

    // 更新内容：将年龄设置为 35
    var update = new BsonDocument("$set", new BsonDocument("age", 35));

    var result = collection.UpdateOne(filter, update);
    Console.WriteLine($"{result.ModifiedCount} document(s) updated.");
}
```

 **说明**：

- `$set` 是 MongoDB 的更新操作符，用于更新字段。
- 可以使用其他更新操作符，例如 `$inc`（自增）、`$unset`（删除字段）。

---
### 使用 BSON 删除数据

通过 `DeleteOne` 或 `DeleteMany` 方法删除数据。

```csharp
public void DeleteUsingBson()
{
    var client = new MongoClient("mongodb://localhost:27017");
    var database = client.GetDatabase("TestDatabase");
    var collection = database.GetCollection<BsonDocument>("Users");

    // 删除条件：删除名字为 "John Doe" 的用户
    var filter = new BsonDocument("name", "John Doe");

    var result = collection.DeleteOne(filter);
    Console.WriteLine($"{result.DeletedCount} document(s) deleted.");
}
```

---
### 使用 BSON 创建索引

为某个字段创建索引（例如 `name` 字段）。

```csharp
public void CreateIndexUsingBson()
{
    var client = new MongoClient("mongodb://localhost:27017");
    var database = client.GetDatabase("TestDatabase");
    var collection = database.GetCollection<BsonDocument>("Users");

    // 创建索引
    var indexKeys = new BsonDocument("name", 1); // 1 表示升序
    collection.Indexes.CreateOne(new CreateIndexModel<BsonDocument>(indexKeys));

    Console.WriteLine("Index created successfully!");
}
```

---
## 完整示例

以下是一个完整的控制台应用程序，展示如何通过 BSON 格式完成基本的增删改查操作：

```csharp
using System;
using MongoDB.Bson;
using MongoDB.Driver;

namespace MongoDBBsonExample
{
    class Program
    {
        static void Main(string[] args)
        {
            var client = new MongoClient("mongodb://localhost:27017");
            var database = client.GetDatabase("TestDatabase");
            var collection = database.GetCollection<BsonDocument>("Users");

            // 插入数据
            var newUser = new BsonDocument
            {
                { "name", "Alice" },
                { "age", 28 },
                { "email", "alice@example.com" }
            };
            collection.InsertOne(newUser);
            Console.WriteLine("User inserted.");

            // 查询数据
            var allUsers = collection.Find(new BsonDocument()).ToList();
            Console.WriteLine("All users:");
            foreach (var user in allUsers)
            {
                Console.WriteLine(user.ToJson());
            }

            // 更新数据
            var filter = new BsonDocument("name", "Alice");
            var update = new BsonDocument("$set", new BsonDocument("age", 30));
            collection.UpdateOne(filter, update);
            Console.WriteLine("User updated.");

            // 删除数据
            collection.DeleteOne(filter);
            Console.WriteLine("User deleted.");
        }
    }
}
```