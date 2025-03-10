在 MongoDB 中，查询时可以通过 `projection` 来指定返回哪些字段。默认情况下，MongoDB 会返回匹配查询条件的文档中的所有字段。为了提高效率，减少不必要的数据传输，通常会选择只返回需要的字段。

类似SQL的

```sql
// 返回name
select name from users
// 返回所有列
select * from users
```

## **基本语法**

在 MongoDB 中，查询时可以在 `find()` 方法的第二个参数中使用一个对象来定义需要返回的字段。该对象是一个“投影”（projection）对象。

```javascript
// projection:指定返回的字段
db.collection.find(query, projection)
```

## **指定返回字段**

返回字段时使用 `1`，排除字段时使用 `0`

```javascript
db.users.find({}, {name: 1})
```

## **注意事项**

你不能在同一个投影中同时使用 1 和 0，除非你只返回 `_id` 字段。比如：

```javascript
db.users.find({}, {name: 1, age: 0})
```

**`_id` 字段**：MongoDB 默认会返回 `_id` 字段。如果你不需要 `_id`，可以显式地将其排除：

```javascript
db.users.find({}, {_id: 0 , name: 1})
```

## **嵌套字段**

如果文档中有嵌套的子文档，可以通过“点”语法来指定嵌套字段。例如，如果文档中有一个 `bag` 字段，它本身是一个对象，包含 `index` ，你可以指定只返回 `index` 字段：

```javascript
db.users.find({}, {"bag.index": 1})
```

## **使用投影来优化性能**

只返回必要的字段，可以减少网络传输的数据量，优化查询性能。尤其是在文档中有很多不必要的数据时，指定字段有助于提高效率。

## **总结**

- 返回字段时使用 `1`，排除字段时使用 `0`。
- 默认返回 `_id` 字段，但可以通过设置 `_id: 0` 来排除。
- 支持嵌套字段的投影，使用点语法来指定。
- 投影是优化查询和减少数据传输的有效手段。