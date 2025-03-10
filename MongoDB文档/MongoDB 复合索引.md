在 MongoDB 中，复合索引（Compound Index）是由多个字段组成的索引，可以帮助优化查询性能，尤其是针对需要同时根据多个字段筛选数据的查询。复合索引使得 MongoDB 可以更高效地查找数据，而不需要扫描整个集合。它的创建方式与单字段索引类似，但是需要指定多个字段，MongoDB 会根据字段顺序来组织索引。
## **语法**

MongoDB 的复合索引是由两个或更多字段的索引组合而成。索引的顺序在查询性能中非常重要，因此复合索引的字段顺序要与查询中使用的条件匹配。复合索引不仅能提高复合查询的性能，还能在某些情况下优化单字段查询。

假设我们有一个名为 `users` 的集合，其中包含字段 `name`、`age` 和 `city`。我们可以创建一个复合索引来优化同时根据 `name` 和 `age` 查询的操作：

```javascript
db.users.createIndex({ name: 1, age: 1 })
```

这样，MongoDB 会根据 `name` 字段升序排列，如果 `name` 相同，则根据 `age` 字段升序排列。此索引将优化类似以下查询：

```javascript
db.users.find({ name: "Alice", age: 30 })
```
## **使用场景**

### **多字段查询**

当查询需要根据多个字段过滤数据时，复合索引非常有用。例如，查询 `name` 和 `age` 的组合字段会比单独使用这两个字段的单索引更有效。
### **排序查询**

复合索引不仅能提高筛选查询的性能，还能加速基于多个字段的排序操作。例如，创建一个按 `name` 和 `age` 排序的复合索引：

```javascript
db.users.createIndex({ name: 1, age: -1 })
```

这样可以有效支持类似以下的查询：

```javascript
db.users.find().sort({ name: 1, age: -1 })
```

### **前缀匹配**

MongoDB 复合索引的字段顺序非常重要，因为查询只会利用索引的前缀部分。例如，如果创建了 `{ name: 1, age: 1, city: 1 }` 的索引，那么只有查询中包含 `name` 和 `age` 的组合，或者仅查询 `name` 的字段时，才能使用该索引，而查询只根据 `city` 字段将无法利用此索引。

复合索引的“前缀部分”是指索引中从左到右**连续的字段**。对于 `{ name: 1, age: 1, city: 1 }` 的索引，前缀部分是：

- `{ name: 1 }`（只使用 `name` 字段）
- `{ name: 1, age: 1 }`（`name` 和 `age` 字段）
- `{ name: 1, age: 1, city: 1 }`（`name`、`age` 和 `city` 字段）

#### **查询包含索引前缀的字段**

- **查询包含 `name` 和 `age` 字段**，比如 `db.users.find({ name: "Alice", age: 30 })`，MongoDB 可以有效地使用 `{ name: 1, age: 1, city: 1 }` 这个复合索引。因为查询条件包含索引中的前两个字段 `name` 和 `age`，它完全匹配索引的前缀部分。
    
- **查询只包含 `name` 字段**，比如 `db.users.find({ name: "Alice" })`，MongoDB 也可以使用该索引，因为 `name` 是索引的第一个字段，查询中包含了该字段。
    
- **查询包含 `name`、`age` 和 `city` 字段**，比如 `db.users.find({ name: "Alice", age: 30, city: "New York" })`，MongoDB 也会使用 `{ name: 1, age: 1, city: 1 }` 索引，因为查询条件完全匹配复合索引的字段顺序。

#### **查询只包含非前缀字段时**

- **查询只包含 `city` 字段**，比如 `db.users.find({ city: "New York" })`，MongoDB 无法有效地利用 `{ name: 1, age: 1, city: 1 }` 索引。这是因为 `city` 是索引中的第三个字段，并且它没有与前面的字段（`name` 和 `age`）一起出现在查询中。MongoDB 只能使用索引的前缀部分，而不支持跳过前缀部分的字段来使用该索引。因此，它会选择全表扫描或其他可能的索引（如果有的话），因为没有按索引顺序匹配的字段。

#### **索引的前缀匹配规则**

复合索引中，查询**必须从索引的左端开始**，并且可以**逐步增加字段**来进行匹配。也就是说，如果索引是 `{ name: 1, age: 1, city: 1 }`，查询可以利用：

- 只查询 `name`
- 查询 `name` 和 `age`
- 查询 `name`、`age` 和 `city`

但是，如果查询中只包含 `age` 或 `city` 字段，或者是从 `age` 或 `city` 开始的字段组合，MongoDB 将无法使用该复合索引，因为它没有按照索引的前缀顺序来匹配字段。
## **复合索引的优化**

MongoDB 会根据查询的字段顺序来选择复合索引。如果你的查询条件与索引顺序不匹配，可能导致不使用索引或降低查询性能。因此，创建复合索引时要考虑常用查询的字段顺序。

**使用 `$and` 连接多个条件**：当查询条件使用 `$and` 运算符连接多个字段时，MongoDB 可能会选择适合的复合索引来优化查询。

例如，创建一个复合索引：

```javascript
db.users.createIndex({ name: 1, age: 1, city: 1 })
```

然后运行如下查询：

```javascript
db.users.find({ name: "Alice", $and: [{ age: 30 }, { city: "New York" }] })
```

## **复合索引的例子**

###  根据多个字段进行查询

假设有一个 `orders` 集合，包含字段 `customerId`、`orderDate` 和 `status`，你想要查询所有特定客户在某个日期之后且状态为已完成的订单：

```javascript
db.orders.createIndex({ customerId: 1, orderDate: 1, status: 1 })
```

这个复合索引可以优化以下查询：

```javascript
db.orders.find({ 
   customerId: 123, 
   orderDate: { $gt: new Date('2024-01-01') }, status: "completed" 
})
```

### 排序查询优化

假设我们有一个 `products` 集合，包含字段 `category`、`price` 和 `stock`。如果你想要查询某个类别的所有商品，并根据价格从高到低排序，你可以创建以下复合索引：

```javascript
db.products.createIndex({ category: 1, price: -1 })
```

这样，以下查询就会更高效：

```javascript
db.products.find({ category: "electronics" }).sort({ price: -1 })
```

## **复合索引的注意事项**

- **索引的大小**：复合索引会占用更多的存储空间，因为它包含多个字段的信息。在索引字段增多时，可能会增加数据库的磁盘占用和更新开销。
- **更新性能**：每次插入、更新或删除文档时，MongoDB 都需要更新与之相关的索引，因此频繁的更新操作可能会导致性能下降。
- **字段顺序**：索引的字段顺序会影响查询性能，尤其是当查询条件只包含复合索引的前缀字段时，索引的顺序非常重要。

### 总结

MongoDB 的复合索引是优化多字段查询和排序的重要工具。选择合适的字段组合和顺序能够显著提高查询性能。创建复合索引时，需谨慎考虑查询模式，避免无效的索引带来的额外开销。









