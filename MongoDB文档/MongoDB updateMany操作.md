`updateMany` 是 MongoDB 中用来批量更新多个文档的操作方法。它根据指定的查询条件匹配多个文档，并对这些文档应用更新操作。常用于需要一次性修改多条记录的场景。
## 语法

```javascript
db.collection.updateMany(
  <filter>,      // 查询条件，用于匹配需要更新的文档
  <update>,      // 更新操作（如 $set, $unset 等）
  <options>      // 可选参数（如 upsert、arrayFilters）
)
```

参数说明：

1. **`<filter>`**
    
    - 用于匹配需要更新的文档。
    - 它是一个查询条件，支持 MongoDB 所有的查询操作符（如 `$eq`, `$gt`, `$or` 等）。
    
1. **`<update>`**
    
    - 定义要对匹配的文档执行的更改。
    - 通常使用更新操作符（如 `$set`, `$unset`, `$inc` 等），也可以使用替换文档（不推荐）。
    
1. **`<options>`**
    
    - **`upsert`**：如果没有匹配到文档，是否插入一个新文档。默认为 `false`。
    - **`arrayFilters`**：用于更新嵌套数组中符合条件的元素（需要 MongoDB 3.6+）。

---
## 场景

1. **批量修改文档的某些字段：**  
    1. 当需要对多条记录进行统一的字段更新时，比如修改多个用户的状态。
    
2. **逻辑字段更新：**  
    1. 比如在某些场景下，需要批量增加计数器或更新标志值。
    
3. **数据清理：**  
    1. 批量删除字段、重置数据或修复错误值。
    
4. **嵌套数组更新：**  
    1. 更新数组内多个元素，或批量移除嵌套字段。

---
## 示例

### 使用 `$set` 更新多个文档的字段值

更新 `status` 字段为 `"active"`，条件是 `age > 25` 的用户。

```javascript
db.users.updateMany(
  { age: { $gt: 25 } }, // 查询条件
  { $set: { status: "active" } } // 更新操作
);
```

### 使用 `$unset` 删除字段

删除所有用户文档中的字段 `tempField`。

```javascript
db.users.updateMany(
  {}, // 匹配所有文档
  { $unset: { tempField: "" } } // 删除字段
);
```

### 使用 `$inc` 增加计数器

将所有库存数量（`stock`）增加 10，条件是 `category` 为 `"electronics"` 的商品。

```javascript
db.products.updateMany(
  { category: "electronics" },
  { $inc: { stock: 10 } }
);
```

### 使用 `upsert` 插入文档

如果没有找到匹配的文档（`type: "new"`），则插入一个新文档。

```javascript
db.items.updateMany(
  { type: "new" },
  { $set: { status: "available", price: 0 } },
  { upsert: true }
);
```

### 更新嵌套数组中的多个字段

在每个文档中，更新 `grades` 数组中 `score` 大于 80 的元素，将它们的 `passed` 字段设为 `true`。

```javascript
db.students.updateMany(
  { "grades.score": { $gt: 80 } }, // 查询条件
  { $set: { "grades.$[elem].passed": true } }, // 更新操作
  { arrayFilters: [{ "elem.score": { $gt: 80 } }] } // 针对数组元素的条件
);
```

---
## 注意事项

1. **必须使用更新操作符：**  
    如果没有使用 `$set`, `$unset` 等更新操作符，而是直接传递一个文档，MongoDB 会将匹配的文档完全替换为新文档。
    
2. **区分 `updateOne` 和 `updateMany`：**
    
    - `updateOne` 只更新第一个匹配的文档。
    - `updateMany` 会更新所有匹配的文档。
3. **`upsert` 的副作用：**
    
    - 如果设置了 `upsert: true`，MongoDB 可能会插入新文档（如果没有匹配到任何文档）。
    - 插入的新文档会基于查询条件和更新内容生成，可能导致意外行为。
4. **数组更新需要 `arrayFilters`：**
    
    - 如果更新嵌套数组中的元素，必须使用 `arrayFilters` 明确指定条件（从 MongoDB 3.6 开始支持）。
5. **性能问题：**
    
    - 批量更新可能会锁定多个文档，尤其是在高并发情况下可能影响性能。
    - 建议在更新前索引查询条件，以提高匹配效率。
6. **字段类型的一致性：**
    
    - 在更新字段时，要确保字段类型的统一性，以避免查询或后续操作中的问题。

---
## 总结

- **`updateMany` 是 MongoDB 中批量更新文档的操作方法，适用于需要一次性修改多条记录的场景。**
- **支持丰富的查询条件和更新操作，可以灵活地对文档或嵌套数组进行更新。**
- **在使用时需要注意性能问题、`upsert` 的副作用以及字段类型的一致性。**
- **常见的更新操作包括 `$set`, `$unset`, `$inc`, `$push`, `$pull` 等，结合数组过滤器可以实现更复杂的操作。**