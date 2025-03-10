MongoDB 的唯一索引（Unique Index）是一种特殊的索引类型，它确保索引字段中的值是唯一的。唯一索引在数据完整性方面非常有用，特别是在需要防止重复数据的场景中，例如用户邮箱、用户名等。
## 特点

1. **唯一性约束**：唯一索引确保在集合中没有两条文档的指定字段值相同。
2. **单字段与复合字段**：唯一索引既可以应用于单个字段，也可以应用于多个字段（复合索引），从而保证这些字段组合的唯一性。
3. **默认 `_id` 唯一索引**：MongoDB 默认会在每个集合的 `_id` 字段上创建唯一索引。
4. **插入/更新限制**：当在某个字段上创建唯一索引后，插入或更新文档时，如果新值违反了唯一性约束，则操作会失败。

---
## 语法

### 基础语法

使用 `db.collection.createIndex()` 方法创建唯一索引：

```javascript
db.collection.createIndex({ field: 1 }, { unique: true })
```

- `{ field: 1 }`：表示索引的字段，`1` 表示升序索引。
- `{ unique: true }`：表示该索引为唯一索引。
### 创建复合唯一索引

复合唯一索引可以确保多个字段的组合值是唯一的：

```javascript
db.collection.createIndex({ field1: 1, field2: 1 }, { unique: true })
```

---
## 示例

以下是一些实际操作的例子，帮助理解如何使用 MongoDB 的唯一索引。
### 单字段唯一索引

在 `users` 集合中，确保 `email` 字段唯一：

```javascript
// 创建唯一索引
db.users.createIndex({ email: 1 }, { unique: true })

// 插入文档
db.users.insertOne({ name: "Alice", email: "alice@example.com" })
// 成功插入

db.users.insertOne({ name: "Bob", email: "alice@example.com" })
// 插入失败，因为 email 重复
```

### 复合唯一索引

在 `orders` 集合中，每个用户的订单编号必须唯一：

```javascript
// 创建复合唯一索引
db.orders.createIndex({ userId: 1, orderNumber: 1 }, { unique: true })

// 插入文档
db.orders.insertOne({ userId: 1, orderNumber: 1001, product: "Laptop" })
// 成功插入

db.orders.insertOne({ userId: 1, orderNumber: 1001, product: "Mouse" })
// 插入失败，因为 userId 和 orderNumber 的组合重复

db.orders.insertOne({ userId: 2, orderNumber: 1001, product: "Keyboard" })
// 成功插入，因为 userId 不同
```

### 尝试在已有重复数据的字段上创建唯一索引

如果集合中已经存在重复数据，则直接创建唯一索引会失败：

```javascript
// 插入重复数据
db.users.insertMany([
  { name: "Alice", email: "alice@example.com" },
  { name: "Bob", email: "alice@example.com" } // 重复 email
])

// 尝试创建唯一索引
db.users.createIndex({ email: 1 }, { unique: true })
// 错误：索引创建失败，因为已经有重复数据
```

### 唯一索引与稀疏索引

稀疏索引允许文档中不包含索引字段的情况。结合唯一索引，可以实现部分文档的唯一性。

```javascript
// 创建稀疏唯一索引
db.users.createIndex({ phone: 1 }, { unique: true, sparse: true })

// 插入文档
db.users.insertOne({ name: "Alice" }) // 成功插入，因为 phone 字段不存在
db.users.insertOne({ name: "Bob", phone: "1234567890" }) // 成功插入
db.users.insertOne({ name: "Eve", phone: "1234567890" }) // 插入失败，phone 重复
```

### 删除唯一索引

```javascript
db.collection.dropIndex("index_name")
```

---
## 注意事项

1. **插入与更新校验**：唯一索引会对插入和更新操作进行校验，确保不会产生违反唯一性约束的文档。
2. **性能影响**：索引的创建和维护会增加写入操作的开销，但能显著提高查询性能。
3. **稀疏索引的使用**：稀疏索引只能应用于部分文档，适合字段可能缺失的场景。
## 总结

MongoDB 的唯一索引（Unique Index）是用来保证某个字段或字段组合的值在集合中是唯一的，是数据完整性的重要工具。以下是本文的核心内容总结：

- **功能特点**：
    
    - 唯一索引可以防止重复数据的插入或更新。
    - 默认情况下，MongoDB 在 `_id` 字段上会自动创建唯一索引。
- **创建方式**：
    
    - 单字段唯一索引：确保单个字段的值唯一。
    - 复合唯一索引：确保多个字段的组合值唯一。
- **操作步骤**：
    
    - 使用 `db.collection.createIndex()` 创建唯一索引。
    - 使用 `db.collection.dropIndex()` 删除索引。
    - 使用 `db.collection.getIndexes()` 查看现有索引。
- **注意事项**：
    
    - 在已存在重复数据的字段上创建唯一索引会失败，需要先清理数据。
    - 唯一索引结合稀疏索引（Sparse Index）可应用于部分文档。
    - 索引的创建和维护会增加写操作的开销，但能提升查询性能。
- **典型场景**：
    
    - 用户账号系统中的邮箱、用户名等字段需要唯一性。
    - 电商订单系统中用户 ID 和订单编号的组合需要唯一性。

通过合理地使用唯一索引，可以有效保证 MongoDB 数据库的完整性，同时减少开发中因数据重复导致的逻辑错误，从而让系统更加可靠。