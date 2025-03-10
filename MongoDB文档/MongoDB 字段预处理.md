MongoDB的聚合操作符 `$project` 是一种强大的工具，用于控制文档的输出内容。通过 `$project`，你可以选择文档中的特定字段、重命名字段、创建新字段、以及对现有字段进行变换。它通常用于对数据进行预处理，返回聚合操作中的定制结果。

## 功能

1. **选择字段**： 你可以使用 `$project` 选择文档中需要保留的字段，或者排除不需要的字段。默认情况下，MongoDB返回文档中的所有字段，但使用 `$project` 后，你可以精确地指定返回哪些字段。
    
2. **重命名字段**： `$project` 可以将现有字段重命名为新的字段名。
    
3. **创建新字段**： 通过 `$project`，可以使用现有字段的数据来创建新的字段。
    
4. **字段计算和变换**： 可以对现有字段进行算术计算、字符串操作、日期转换等操作。
    
5. **排除字段**： 除了选择字段外，`$project` 还可以用于排除某些字段（通常通过将字段设置为 0 来排除）。

## 语法

```javascript
db.collection.aggregate([
  {
    $project: {
      field1: 1,        // 包含字段 field1
      field2: 0,        // 排除字段 field2
      newField: { $add: ["$field1", 10] },  // 创建新字段 newField，它的值是 field1 的值加 10
      renamedField: "$field1"  // 将 field1 重命名为 renamedField
    }
  }
])
```

## 常用的操作

- **选择字段**：通过设定值为 `1` 来选择字段，设定为 `0` 来排除字段。例如：
    
    - `field1: 1` 表示保留 `field1` 字段。
    - `field2: 0` 表示排除 `field2` 字段。
    
- **创建新字段**：你可以使用各种运算符来生成新字段，例如 `$add`（加法）、`$subtract`（减法）、`$multiply`（乘法）、`$concat`（字符串连接）等。例如：
    
    - `totalPrice: { $multiply: ["$quantity", "$price"] }`，计算 `quantity` 和 `price` 的乘积作为新字段 `totalPrice`。
    
- **重命名字段**：通过 `$project` 可以将字段的名称进行更改。例如：
    
    - `newName: "$oldName"`，将 `oldName` 字段重命名为 `newName`。

## 示例

### 准备操作

为了方便例子的演示，需要将如下数据插入到数据库`orders`中。

```json
db.orders.insertMany([
  {
    "orderId": 1,
    "customerId": 626,
    "quantity": 12,
    "price": 26,
    "discount": 0.17,
    "orderDate": ISODate("2024-05-04T00:00:00Z")
  },
  {
    "orderId": 2,
    "customerId": 798,
    "quantity": 8,
    "price": 31,
    "discount": 0.09,
    "orderDate": ISODate("2024-03-07T00:00:00Z")
  },
  {
    "orderId": 3,
    "customerId": 531,
    "quantity": 13,
    "price": 50,
    "discount": 0.23,
    "orderDate": ISODate("2024-10-03T00:00:00Z")
  }
])
```

### 示例 1：只选 `orderId` 和 `totalPrice`

```javascript
db.orders.aggregate([
  {
    $project: {
      _id: 0,
      orderId: 1,  // 保留 orderId
      totalPrice: { $multiply: ["$quantity", "$price"] }  // 计算 totalPrice
    }
  }
])
```

**结果：**

```javascript
[
  { orderId: 1, totalPrice: 312},
  { orderId: 2, totalPrice: 248},
  { orderId: 3, totalPrice: 650}
]
```

### 示例 2：重命名字段并进行计算

假设我们希望将 `quantity` 字段重命名为 `amount`，并且计算每个订单的总价格 `total`。

```javascript
db.orders.aggregate([
  {
    $project: {
      _id: 0,
      amount: "$quantity",  // 将 quantity 重命名为 amount
      total: { $multiply: ["$quantity", "$price"] }  // 计算 total 价格
    }
  }
])
```

**结果：**

```json
[
  { "amount": 12, "total": 312 },
  { "amount": 8, "total": 248 },
  { "amount": 13, "total": 650 }
]
```

### 示例 3：排除某些字段并添加新字段

假设我们不需要返回 `customerId`，并且我们还想计算订单的折扣价（假设有一个字段 `discount`）。

```javascript
db.orders.aggregate([
  {
    $project: {
      _id: 0,
      orderId: 1,  // 保留 orderId
      quantity: 1,  // 保留 quantity
      price: 1,  // 保留 price
      discountPrice: { $multiply: ["$price", { $subtract: [1, "$discount"] }] }  // 计算折扣价
    }
  }
])
```

**结果：**

```json
[
  { "orderId": 1, "quantity": 12, "price": 26, "discountPrice": 21.58 },
  { "orderId": 2, "quantity": 8, "price": 31, "discountPrice": 28.21 },
  { "orderId": 3, "quantity": 13, "price": 50, "discountPrice": 38.5 }
]
```

### 示例 4：对日期字段进行变换

如果文档中包含一个日期字段 `orderDate`，你可以使用 `$project` 结合日期操作符来提取年、月、日。

```javascript
db.orders.aggregate([
  {
    $project: {
      _id: 0,
      orderId: 1,
      year: { $year: "$orderDate" },  // 提取年份
      month: { $month: "$orderDate" },  // 提取月份
      day: { $dayOfMonth: "$orderDate" }  // 提取日期
    }
  }
])
```

**结果：**

```javascript
[
  { "orderId": 1, "year": 2024, "month": 5, "day": 4 },
  { "orderId": 2, "year": 2024, "month": 3, "day": 7 },
  { "orderId": 3, "year": 2024, "month": 10, "day": 3 }
]
```

## 总结

`$project` 是一个非常强大的聚合操作符，它可以帮助你控制结果文档的字段内容，进行数据处理、字段重命名、计算新字段等操作。通过合理使用 `$project`，你可以优化数据的呈现方式，避免不必要的数据传输和复杂的查询逻辑











