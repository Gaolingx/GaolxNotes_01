`$group` 操作符是 MongoDB 聚合管道中的一个关键组成部分，类似于 SQL 中的 `GROUP BY`，主要用于将文档根据指定字段进行分组，并对每个组执行聚合计算。以下是对 `$group` 操作符的详细介绍，包括其功能、用法、常用聚合操作符、以及实际应用示例。

## 功能

- **分组文档**：`$group` 将输入文档按指定字段的值进行分组。每个唯一值形成一个组。
- **聚合计算**：对于每个组，可以使用聚合表达式计算汇总信息，例如计数、求和、计算平均值等。

## 基本语法

```javascript
{
    $group: {
        _id: <expression>,            // 需要分组的字段
        <field1>: { <accumulator1> }, // 计算字段和对应的聚合操作
        <field2>: { <accumulator2> }  // 其他计算字段
    }
}
```

- `_id` 字段定义了分组的依据。可以是文档中的某个字段，也可以是计算得出的表达式。
- `<accumulator>` 是聚合操作符，用于计算每个组的值。

## 常用聚合操作符

`$group` 中常用的聚合操作符包括：

- **`$sum`**：计算总和。
- **`$avg`**：计算平均值。
- **`$max`**：获取组中最大值。
- **`$min`**：获取组中最小值。
- **`$first`**：获取组中第一个文档的字段值。
- **`$last`**：获取组中最后一个文档的字段值。
- **`$push`**：将字段值添加到一个数组中。
- **`$addToSet`**：将字段值添加到一个集合中，集合中的值唯一。

## 例子

### 准备数据

首先准备一些数据供下面使用

```javascript
db.sales.insertMany([
	{ "item": "apple", "quantity": 10, "price": 1.5 },
	{ "item": "banana", "quantity": 5, "price": 0.5 },
	{ "item": "apple", "quantity": 8, "price": 1.5 },
	{ "item": "orange", "quantity": 12, "price": 1.0 },
	{ "item": "banana", "quantity": 7, "price": 0.5 }
])
```

### **`$sum`**

计算每种商品的总销售数量。

```javascript
db.sales.aggregate([
    {
        $group: {
            _id: "$item",
            totalQuantity: { $sum: "$quantity" }
        }
    }
])
```

**结果**：

```javascript
[
    { "_id": "apple", "totalQuantity": 18 },
    { "_id": "banana", "totalQuantity": 12 },
    { "_id": "orange", "totalQuantity": 12 }
]
```

### **`$avg`**

计算每种商品的平均价格。

```javascript
db.sales.aggregate([
    {
        $group: {
            _id: "$item",
            averagePrice: { $avg: "$price" }
        }
    }
])
```

**结果**：

```javascript
[
    { "_id": "apple", "averagePrice": 1.5 },
    { "_id": "banana", "averagePrice": 0.5 },
    { "_id": "orange", "averagePrice": 1.0 }
]
```

### **`$max`**

获取每种商品的最高销售数量。

```javascript
db.sales.aggregate([
    {
        $group: {
            _id: "$item",
            maxQuantity: { $max: "$quantity" }
        }
    }
])
```

**结果**：

```javascript
[
    { "_id": "apple", "maxQuantity": 10 },
    { "_id": "banana", "maxQuantity": 7 },
    { "_id": "orange", "maxQuantity": 12 }
]
```

### `$min`

获取每种商品的最低销售数量。

```javascript
db.sales.aggregate([
    {
        $group: {
            _id: "$item",
            minQuantity: { $min: "$quantity" }
        }
    }
])
```

**结果**：

```javascript
[
    { "_id": "apple", "minQuantity": 8 },
    { "_id": "banana", "minQuantity": 5 },
    { "_id": "orange", "minQuantity": 12 }
]
```

### `$first`

获取每种商品的第一个销售记录的数量。

```javascript
db.sales.aggregate([
    {
        $group: {
            _id: "$item",
            firstQuantity: { $first: "$quantity" }
        }
    }
])
```

**结果**：

```javascript
[
    { "_id": "apple", "firstQuantity": 10 },
    { "_id": "banana", "firstQuantity": 5 },
    { "_id": "orange", "firstQuantity": 12 }
]
```

### `$last`

获取每种商品的最后一个销售记录的数量。

```javascript
db.sales.aggregate([
    {
        $group: {
            _id: "$item",
            lastQuantity: { $last: "$quantity" }
        }
    }
])
```

**结果**：

```javascript
[
    { "_id": "apple", "lastQuantity": 8 },
    { "_id": "banana", "lastQuantity": 7 },
    { "_id": "orange", "lastQuantity": 12 }
]
```

### `$push`

将每种商品的所有销售数量放入一个数组中。

```javascript
db.sales.aggregate([
    {
        $group: {
            _id: "$item",
            quantities: { $push: "$quantity" }
        }
    }
])
```

**结果**：

```javascript
[
    { "_id": "apple", "quantities": [10, 8] },
    { "_id": "banana", "quantities": [5, 7] },
    { "_id": "orange", "quantities": [12] }
]
```

### `$addToSet`

将每种商品的所有销售数量放入一个集合中（唯一值）。

```javascript
db.sales.aggregate([
    {
        $group: {
            _id: "$item",
            uniqueQuantities: { $addToSet: "$quantity" }
        }
    }
])
```

**结果**：

```javascript
[
    { "_id": "apple", "uniqueQuantities": [10, 8] },
    { "_id": "banana", "uniqueQuantities": [5, 7] },
    { "_id": "orange", "uniqueQuantities": [12] }
]
```











