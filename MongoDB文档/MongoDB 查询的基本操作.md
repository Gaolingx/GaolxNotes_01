MongoDB 使用 `find` 方法进行查询，常见形式如下：

```javascript
db.collection.find(query, projection)
```

- **`query`**: 查询条件，指定筛选数据的规则。
- **`projection`**: 投影，用于指定返回哪些字段（1 表示包含，0 表示排除）。

## 查询所有文档

```javascript
db.users.find()
```

## 查询带条件的文档

### 等于

```javascript
// 查询年龄为 25 的用户
db.users.find({ age: 25 })
```

### 不等于

```javascript
// 查询年龄不等于 25 的用户
db.users.find({ age: { $ne: 25 } })
```

### 大于

```javascript
// 查询年龄大于 25 的用户
db.users.find({ age: { $gt: 20 } })
```

### 小于

```javascript
// 查询年龄小于 25 的用户
db.users.find({ age: { $lt: 30 } })
```

### 大于等于

```javascript
// 查询年龄大于等于 25 的用户
db.users.find({ age: { $gte: 25 } })
```

### 小于等于

```javascript
// 查询年龄小于等于 25 的用户
db.users.find({ age: { $lte: 25 } })
```


## 查询字段条件

### 字段是否存在

```javascript
// 查看是有age这个字段的所有数据
db.users.find({ age: { $exists: true } })
```
### 字段类型匹配

```javascript
db.users.find({ age: { $type: "int" } })
```

## 查询数组里的元素

### 插入一个带数组的文档

```javascript
// 插入一个新的数据，并且增加一个Lessons数组字段
db.users.insertOne({ 
	name: "Fantasy1", 
	age: 25, 
	lessons: [1, 3, 5, 7, 9] 
});
db.users.insertOne({ 
	name: "Fantasy1", 
	age: 25, 
	lessons: [2, 4, 6, 8, 10] 
});
```

### 查询在指定数组里的值

```javascript
// 查询lessons里有1或4的元素的数据
// 这里查询的是类似或的操作，不是和的操作
db.users.find({ 
	lessons: { $in: [1, 4] } 
});
```

### 查询不在指定数据里的值

```javascript
// 查询lessons里有1或4的元素的数据
db.users.find({ 
	lessons: { $nin: [11, 4] } 
});
```

### 查询数组中符合条件的值

```javascript
// 查询lessons里大于2的元素
db.users.find({ 
	lessons: { $elemMatch: { $gt: 2 } } 
});
```

## 嵌套字段的查询

### 插入一个带嵌套字段的数据

```javascript
// 插入两条带bag对象的数据
db.users.insertOne({ 
	name: "Fantasy1", 
	age: 25, 
	bag: { 
		index: 1 
		} 
});
db.users.insertOne({ 
	name: "Fantasy1", 
	age: 25, 
	bag: { 
		index: 2 
		} 
});
```

###  **查询 `bag.index` 的确切值**

```javascript
db.users.find({ "bag.index": 1 });
```

### **查询 `bag.index` 的范围**


```javascript
db.users.find({ "bag.index": { $gt: 0 } });
```