**部分索引（Partial Index）** 是 MongoDB 提供的一种优化索引的方法，用于只索引满足特定条件的文档。通过只为部分文档创建索引，可以减少索引的大小，提高查询性能，同时节省存储空间。

---
## 语法

```javascript
db.collection.createIndex([
   { field1: 1 }, // 索引字段
   { partialFilterExpression: { field2: { $gte: 10 } } } // 过滤条件
]);
```

---
## 应用场景

1. **稀疏数据**：当集合中有很多文档某些字段值为空或不存在时，为这些字段建立部分索引可以避免无意义的索引条目。
2. **特定查询优化**：当某些查询只针对特定条件的文档时，部分索引可以显著提高查询效率。
3. **节省存储资源**：部分索引比普通索引占用更少的存储空间。

---
## 示例

### 为特定条件创建部分索引

假设有一个 `orders` 集合，包含以下文档：

```javascript
{ "_id": 1, "status": "completed", "amount": 500 }
{ "_id": 2, "status": "pending", "amount": 200 }
{ "_id": 3, "status": "completed", "amount": 700 }
{ "_id": 4, "status": "canceled", "amount": 300 }
```

如果我们只关心 `status` 为 `"completed"` 的文档，可以创建部分索引：

```javascript
db.orders.createIndex(
  { amount: 1 }, // 按 `amount` 字段排序的索引
  { partialFilterExpression: { status: "completed" } } // 仅为 `status` 为 "completed" 的文档创建索引
);
```

这样，索引只会包含以下两个文档：

```javascript
{ "_id": 1, "status": "completed", "amount": 500 }
{ "_id": 3, "status": "completed", "amount": 700 }
```

查询优化示例：

```javascript
db.orders.find({ status: "completed", amount: { $gt: 600 } });
```

该查询会使用部分索引，效率更高。

---
### 为稀疏字段创建部分索引

假设有一个 `users` 集合，包含以下文档：

```javascript
{ "_id": 1, "name": "Alice", "email": "alice@example.com" }
{ "_id": 2, "name": "Bob" }
{ "_id": 3, "name": "Charlie", "email": "charlie@example.com" }
{ "_id": 4, "name": "Dave" }
```

如果我们希望为 `email` 字段创建索引，但该字段在某些文档中不存在，可以使用部分索引：

```javascript
db.users.createIndex(
  { email: 1 }, // 按 `email` 字段排序的索引
  { partialFilterExpression: { email: { $exists: true } } } // 仅为存在 `email` 字段的文档创建索引
);
```

这样，索引只会包含以下两个文档：

```javascript
{ "_id": 1, "name": "Alice", "email": "alice@example.com" }
{ "_id": 3, "name": "Charlie", "email": "charlie@example.com" }
```

查询优化示例：

```javascript
db.users.find({ email: "alice@example.com" });
```

---
### 结合范围条件使用部分索引

假设有一个 `products` 集合，包含以下文档：

```javascript
{ "_id": 1, "name": "Product A", "price": 100, "stock": 50 }
{ "_id": 2, "name": "Product B", "price": 200, "stock": 0 }
{ "_id": 3, "name": "Product C", "price": 300, "stock": 30 }
{ "_id": 4, "name": "Product D", "price": 400, "stock": 0 }
```

如果我们希望优化查询 `stock > 0` 且按 `price` 排序的场景，可以创建部分索引：

```javascript
db.products.createIndex(
  { price: 1 }, // 按 `price` 字段排序的索引
  { partialFilterExpression: { stock: { $gt: 0 } } } // 仅为 `stock > 0` 的文档创建索引
);
```

这样，索引只会包含以下两个文档：

```javascript
{ "_id": 1, "name": "Product A", "price": 100, "stock": 50 }
{ "_id": 3, "name": "Product C", "price": 300, "stock": 30 }
```

查询优化示例：

```javascript
db.products.find({ stock: { $gt: 0 } }).sort({ price: 1 });
```

---
## 稀疏索引 vs 部分索引

MongoDB **不允许**在同一个索引中同时启用 `sparse` 和 `partialFilterExpression`，因为两者的逻辑是重复的，且 `partialFilterExpression` 已经足够灵活，完全可以覆盖稀疏索引的功能。尝试以下操作会导致错误。

### 同时创建

同时启用 `sparse` 和 `partialFilterExpression`

```javascript
db.collection.createIndex(
  { field1: 1 },
  {
    sparse: true,
    partialFilterExpression: { field1: { $exists: true } }
  }
);
```

结果：报错

```javascript
Cannot specify both sparse and partialFilterExpression in an index
```

### 先创建部分索引，再创建稀疏索引

先创建部分索引

```javascript
db.users.createIndex(
  { email: 1 },
  { partialFilterExpression: { email: { $exists: true } } }
);
```

- 该索引只包含 `email` 字段存在的文档，即 `_id: 1` 和 `_id: 3`。
- 索引会应用到查询中包含 `email: { $exists: true }` 的条件。

再创建稀疏索引

```javascript
db.users.createIndex(
  { email: 1 },
  { sparse: true }
);
```

MongoDB **会允许创建这个索引**，它被视为一个新的索引，尽管它的逻辑行为与部分索引非常相似。

结果

- 集合中会存在两个独立的索引：
    
    1. 一个基于 `partialFilterExpression: { email: { $exists: true } }`。
    2. 一个基于 `sparse: true`。
    
- 这两个索引的作用范围相同，但它们的创建方式不同，因此 MongoDB 不会合并它们。

### 部分索引和稀疏索引的优先级

当查询可以匹配多个索引时，**查询优化器（Query Planner）** 会决定使用哪个索引。优化器根据索引的**选择性（Selectivity）**和**查询代价（Cost）**来选择最优的索引。以下是详细说明，以及在部分索引（Partial Index）和稀疏索引（Sparse Index）同时存在时，哪个索引会被优先使用的情况。
#### 查询条件与部分索引完全匹配

如果查询条件与部分索引的 `partialFilterExpression` 完全匹配或更具体，**部分索引会优先被选择**。

```javascript
// 创建部分索引
db.collection.createIndex(
  { email: 1 },
  { partialFilterExpression: { email: { $exists: true } } }
);

// 创建稀疏索引
db.collection.createIndex(
  { email: 1 },
  { sparse: true }
);

// 查询条件
db.collection.find({ email: { $exists: true } });
```

**优化器行为**：

- 查询条件完全匹配 `partialFilterExpression`。
- **部分索引优先被使用**，因为它的过滤条件与查询条件完全一致。
#### 查询条件与稀疏索引更匹配

如果查询条件只检查字段是否存在（`$exists: true`），而部分索引的过滤条件包含额外的逻辑，稀疏索引可能会被选择。

```javascript
// 创建部分索引（更复杂的过滤条件）
db.collection.createIndex(
  { email: 1 },
  { partialFilterExpression: { email: { $exists: true, $ne: null } } }
);

// 创建稀疏索引
db.collection.createIndex(
  { email: 1 },
  { sparse: true }
);

// 查询条件
db.collection.find({ email: { $exists: true } });
```

**优化器行为**：

- 查询条件与稀疏索引完全匹配。
- **稀疏索引可能优先被使用**，因为它代价更低（无需处理额外的过滤条件）。

---
## 注意事项

1. **查询条件必须匹配部分索引的过滤条件**：如果查询条件不匹配 `partialFilterExpression`，索引将不会被使用。
2. **不适合频繁变更的字段**：如果过滤条件中涉及的字段值频繁变化，会增加索引维护成本。
3. **与普通索引互斥**：部分索引不包含集合中所有文档，如果有其他查询需要包含未被部分索引覆盖的文档，应创建额外的普通索引。

## 总结

MongoDB 的**部分索引**通过限制索引范围，可以有效节省存储空间并优化特定场景的查询性能。它适用于稀疏字段、特定条件优化和范围查询等场景。合理设计部分索引，可以在性能和存储成本之间取得平衡。