MongoDB 提供了多种方式来删除文档或集合中的数据，主要包括以下操作：

1. **`deleteOne`** - 删除单个符合条件的文档。
2. **`deleteMany`** - 删除所有符合条件的文档。
3. **`findOneAndDelete`** - 查找并删除单个文档，返回被删除的文档。
4. **`drop`** - 删除整个集合。
5. **`db.collection.remove`** - 老版本的删除方法（已过时，不建议使用）。

## deleteOne

### 语法

```javascript
db.collection.deleteOne(filter, options)
```

- **`filter`**: 查询条件，指定要删除文档的条件。
- **`options`**: 可选参数，目前仅支持 `collation`（指定排序规则）。

### 应用场景

- 当你只需要删除符合条件的第一个文档时使用。

#### **注意事项**

- 如果有多个文档符合条件，只有第一个被删除。
- 删除操作无法撤销，操作前需确保条件准确。

### 示例

```javascript
// 删除 name 为 "Alice" 的第一个文档
db.users.deleteOne({ name: "Alice" });
```

### 文档示例：

```javascript
{ _id: 1, name: "Alice", age: 25 }
{ _id: 2, name: "Alice", age: 30 }
{ _id: 3, name: "Bob", age: 20 }
```

结果：

```javascript
{ _id: 1, name: "Alice", age: 25 } // 被删除
```

---
## deleteMany

### 语法

```javascript
db.collection.deleteMany(filter, options)
```

- **`filter`**: 查询条件，指定要删除的文档条件。
- **`options`**: 可选参数，目前仅支持 `collation`。

### **应用场景**

- 当需要批量删除符合条件的多个文档时使用。

### **注意事项**

- 如果条件为空 `{}`，将删除集合中的所有文档。
- 在批量删除前需谨慎检查条件。

### **示例**

```javascript
// 删除所有 name 为 "Alice" 的文档
db.users.deleteMany({ name: "Alice" });
```

文档示例：

```javascript
{ _id: 1, name: "Alice", age: 25 }
{ _id: 2, name: "Alice", age: 30 }
{ _id: 3, name: "Bob", age: 20 }
```

结果：

```javascript
{ _id: 1, name: "Alice", age: 25 } // 被删除
{ _id: 2, name: "Alice", age: 30 } // 被删除
```

---
## findOneAndDelete

### 语法

```javascript
db.collection.findOneAndDelete(filter, options)
```

- **`filter`**: 查询条件，指定要删除的文档。
- **`options`**: 可选参数，例如：
    - `sort`: 指定排序规则。
    - `projection`: 指定返回的字段。

### **应用场景**

- 当需要删除某个文档并返回被删除的文档内容时使用。

### **注意事项**

- 此操作返回被删除的文档，便于后续操作。
- 如果有多个文档符合条件，可通过 `sort` 指定优先级。

### **示例**

```javascript
// 删除 age 最小的文档并返回
db.users.findOneAndDelete({}, { sort: { age: 1 } });
```

文档示例：

```javascript
{ _id: 1, name: "Alice", age: 25 }
{ _id: 2, name: "Alice", age: 30 }
{ _id: 3, name: "Bob", age: 20 }
```

**结果：**  
返回被删除的文档：

```javascript
{ _id: 3, name: "Bob", age: 20 } // 被删除并返回
```

---
## drop

### 语法

```javascript
db.collection.drop()
```

### **应用场景**

- 当需要删除整个集合时使用。

### **注意事项**

- 此操作会直接删除整个集合，且无法恢复。
- 删除集合后，索引也会随之删除。

### 示例

```javascript
// 删除 users 集合
db.users.drop();
```

**结果：**  
集合 `users` 被完全删除。

---
## **`db.collection.remove`** （已过时）

老版本的删除方法，MongoDB 4.0+ 已不推荐使用。

---
## **总结**

| 操作                 | 用途             | 是否批量删除 | 是否返回删除的文档 |
| ------------------ | -------------- | ------ | --------- |
| `deleteOne`        | 删除单个符合条件的文档    | 否      | 否         |
| `deleteMany`       | 删除所有符合条件的文档    | 是      | 否         |
| `findOneAndDelete` | 查找并删除单个符合条件的文档 | 否      | 是         |
| `drop`             | 删除整个集合         | 是（全部）  | 否         |
| `remove`           | 老版本操作，不建议使用    | 可选     | 否         |

#### **注意事项总结**

1. 删除操作无法撤销，请确保条件准确。
2. 批量删除（`deleteMany` 或空条件）需非常谨慎，避免误删数据。
3. 使用 `findOneAndDelete` 可以更安全地获取被删除的数据。
4. 删除整个集合（`drop`）前，确认集合中没有需要保留的数据。