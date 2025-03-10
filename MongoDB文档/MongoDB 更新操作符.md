MongoDB 提供了一系列更新操作符，用于在更新文档时执行特定的操作，比如修改字段值、数组操作、删除字段等。常用的操作符包括：

- **字段更新操作符**  
    `$set`, `$unset`, `$rename`, `$inc`, `$mul`, `$min`, `$max`
    
- **数组更新操作符**  
    `$push`, `$pull`, `$pop`, `$addToSet`, `$each`, `$slice`, `$sort`
    
- **更新嵌套字段**  
    通过点符号（`.`）访问嵌套字段。
    
- **其他操作符**  
    `$currentDate`, `$setOnInsert`, `$bit`

---
## 字段更新

### **`$set`**: 设置字段的值。

```javascript
db.collection.updateOne({ name: "Alice" }, { $set: { age: 30 } })
```
### **`$unset`**: 删除字段。

```javascript
db.collection.updateOne({ name: "Alice" }, { $unset: { age: "" } })
```
### **`$inc`**: 增加或减少字段值。

```javascript
db.collection.updateOne({ name: "Alice" }, { $inc: { age: 1 } })
```
### **`$mul`**: 字段值乘以指定值。

```javascript
db.collection.updateOne({ name: "Alice" }, { $mul: { salary: 1.1 } })
```
### **`$rename`**: 重命名字段。

```javascript
db.collection.updateOne({ name: "Alice" }, { $rename: { age: "yearsOld" } })
```

---
## 数组更新

### **`$push`**: 添加元素到数组末尾

```javascript
db.collection.updateOne({ name: "Alice" }, { $push: { tags: "new" } })
```
### **`$pull`**: 删除数组中匹配的元素。

```javascript
db.collection.updateOne({ name: "Alice" }, { $pull: { tags: "old" } })
```
### **`$addToSet`**: 添加元素到数组（如果不存在）。

```javascript
db.collection.updateOne({ name: "Alice" }, { $addToSet: { tags: "unique" } })
```
### **`$pop`**: 删除数组的第一个或最后一个元素。

```javascript
db.collection.updateOne({ name: "Alice" }, { $pop: { tags: 1 } }) // 删除最后一个
```
### **`$each`**: 批量添加元素。

```javascript
db.collection.updateOne(
    { name: "Alice" },
    { $push: { tags: { $each: ["tag1", "tag2"], $sort: 1 } } }
)
```

---
## 注意事项

1. **更新操作符与替换操作的区别**:  
    使用更新操作符（如 `$set`）时只更新指定字段，而直接赋值会替换整个文档。
    
2. **`upsert` 参数**:  
    如果设为 `true`，当没有匹配文档时会插入新文档。
    
3. **嵌套字段操作**:  
    使用点符号（`.`）可以更新嵌套字段。
    
4. **数组操作的注意点**:  
    数组操作符如 `$push` 和 `$pull` 会直接操作数组字段，使用时需确保字段类型为数组。
    
5. **性能问题**:  
    更新操作会触发写操作，批量更新应避免影响性能。

---
## 总结

MongoDB 提供了灵活的更新操作方法和操作符，可以满足各种增量更新、字段操作和数组操作的需求。以下是总结：

- **更新方法**: `updateOne`, `updateMany`, `replaceOne`
- **字段更新操作符**: `$set`, `$unset`, `$inc`, `$rename`, `$mul`
- **数组更新操作符**: `$push`, `$pull`, `$pop`, `$addToSet`
- **注意事项**: 区分操作符与替换操作，合理使用 `upsert`，关注性能问题。