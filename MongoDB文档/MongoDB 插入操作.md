## insertOne（插入单条数据）

insertOne 命令用于在一个 MongoDB 集合中插入单个文档。

如果插入成功，数据库会将新文档存储到集合中，并返回一个包含插入结果的响应。

---
### 语法

```javascript
db.collection.insertOne(document, options)
```

- **`db`**: 代表当前数据库。
- **`collection`**: 目标集合的名称
- **`document`**: 需要插入的文档，格式为 JSON 对象。
- **`options`**: 可选参数，用于配置插入操作的行为。

---
### 参数说明

#### document

  这是插入到集合中的数据。
  
  如果未显式指定 `_id` 字段，MongoDB 会自动为文档生成一个唯一的 ObjectId 作为 `_id`。
  
  示例：

```javascript
{ name: "Fantasy", age: 25, city: "BeiJing" }
```
####  options

   writeConcern: 定义写入操作的确认级别（如 `w`、`j`、`wtimeout` 等）。
   
   bypassDocumentValidation: 布尔值，指定是否跳过文档验证（默认为 `false`）。

   示例：

```javascript
{ 
	writeConcern: 
		 { w: "majority", wtimeout: 5000 }, 
		 bypassDocumentValidation: true 
 }
```

---
#### **返回值**

成功执行后，`insertOne` 返回一个包含以下字段的对象：

- **`acknowledged`**: 布尔值，表示操作是否成功。
- **`insertedId`**: 插入文档的 `_id` 值。

示例返回值：

```javascript
{   acknowledged: true, insertedId: ObjectId("6482b6c7e3f4b5e0d4c88e9a") }
```

--- 
### 示例

#### 基本插入文档

```javascript
db.users.insertOne({ name: "Fantasy1", age: 25 });
```
#####  结果

```javascript
{
  acknowledged: true,
  insertedId: ObjectId("6482b6c7e3f4b5e0d4c88e9a")
}
```
#### 插入带 `_id` 的文档

```javascript
db.users.insertOne({ _id: 1, name: "Fantasy2", age: 30 });
```
##### 结果

```javascript
{ acknowledged: true, insertedId: 1 }
```

#### 使用选项配置插入

```javascript
db.users.insertOne(
  { name: "Fantasy3", age: 35 },
  { writeConcern: { w: "majority", wtimeout: 5000 } }
);
```

---
## insertMany（插入多条数据） 

insertMany 命令用于在一个 MongoDB 集合中插入多个文档。

如果插入成功，数据库会将新文档存储到集合中，并返回多个包含插入结果的响应。

### 语法

```javascript
db.collection.insertMany([ 
	{ 文档1 }, 
	{ 文档2 }, 
	... 
], { options })
```

### 示例

假设有一个名为 `students` 的集合，我们想插入多条学生记录：

```javascript
db.students.insertMany([ 
	{ name: "Fantasy4", age: 20 }, 
	{ name: "Fantasy5", age: 22 }, 
	{ name: "Fantasy6", age: 19 } 
])
```

**结果**：

```javascript
{ 
	"acknowledged": true, 
	"insertedIds": [ 
		ObjectId("64b645cd1f7e4c25fc00123d"), 
		ObjectId("64b645cd1f7e4c25fc00123e"), 
		ObjectId("64b645cd1f7e4c25fc00123f")
	 ] 
 }
```

## 错误处理

##### 重复 _id 错误

如果插入的文档的 `_id` 已存在，MongoDB 会抛出错误：

```javascript
MongoServerError: E11000 duplicate key error collection: test.users index: _id_ dup key: { _id: 1 }
```
