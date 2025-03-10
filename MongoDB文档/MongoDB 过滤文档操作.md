`$match` 是 MongoDB 聚合框架中的一个重要阶段，用于根据指定的条件过滤文档。它的作用与普通的查询比较类似，但可以与其他聚合管道操作结合使用，以实现更复杂的数据处理。

## 基本语法

```javascript
{ $match: { <query> } }
```

其中 `<query>` 是一个用来指定筛选条件的查询对象。这个查询对象可以包含各种条件操作符，用于精确匹配文档。

## 常用的条件操作符

- **基本比较操作符**：
    
    - `$eq`: 等于
    - `$ne`: 不等于
    - `$gt`: 大于
    - `$gte`: 大于或等于
    - `$lt`: 小于
    - `$lte`: 小于或等于
    
- **逻辑操作符**：
    
    - `$and`: 与
    - `$or`: 或
    - `$not`: 非
    - `$nor`: 不满足
    
- **数组操作符**：
    
    - `$in`: 包含于
    - `$nin`: 不包含于
    
- **正则表达式**：
    
    - 可以通过字符串方式进行模式匹配。

## 示例

下面通过几个具体的例子来演示 `$match` 的用法

### 准备数据

```javascript
db.users1.insertMany([
   { "_id": 1, "name": "李雷", "age": 30, "city": "北京" },
   { "_id": 2, "name": "韩梅梅", "age": 22, "city": "上海" },
   { "_id": 3, "name": "赵琳", "age": 25, "city": "广州" },
])
```

### 查询所有年龄大于 25 的用户

```javascript
db.users1.aggregate([
   { $match: { age: { $gt: 25 } } }
])
```

### 查询年龄在 20 到 30 之间的用户

```javascript
db.users1.aggregate([
   { $match: { age: { $gte: 20, $lte: 30 } } }
])
```

### 查询来自 "上海" 且年龄大于 25 的用户

```javascript
db.users1.aggregate([
   { 
      $match: { 
         city: "北京", 
         age: { $gt: 25 } 
      } 
   }
])
```

### 查询年龄大于 30 或者来自 "上海" 的用户

```javascript
db.users1.aggregate([
   { 
      $match: { 
         $or: [
            { age: { $gt: 30 } },
            { city: "上海" }
         ] 
      } 
   }
])
```

### 查询名字以 "李" 开头的用户

```javascript
db.users1.aggregate([
   { $match: { name: { $regex: /^李/ } } }
])
```

## 注意事项

- **性能**：在聚合管道中，尽早使用 `$match` 可以提高性能，因为它会减少后续处理阶段的数据量。
    
- **排序**：`$match` 可以与 `$sort` 一起使用，但通常在 `$match` 后进行 `$sort` 可以获得更好的性能，特别是在集合较大时。
    
- **结合其他操作**：`$match` 可以与其他聚合操作如 `$group`, `$project`, `$sort` 等结合使用，以实现更复杂的数据分析。



