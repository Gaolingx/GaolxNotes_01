
MongoDB 的文本索引 (Text Index) 是一种特殊类型的索引，它允许我们对字符串类型的字段进行全文搜索。通过创建文本索引，MongoDB 可以在大量文本数据中执行高效的文本搜索操作。这个特性特别适用于需要对大量文本数据进行查询的场景，如博客、文章、评论等内容管理系统。

## 语法

文本索引允许在一个或多个字段上创建索引，支持对这些字段进行全文搜索。你可以通过 `createIndex` 方法来创建文本索引。下面是创建文本索引的基本示例

```javascript
// 单字段索引
db.collection.createIndex({ title: "text" })
// 复合索引
db.collection.createIndex({ title: "text", content: "text" })
```

这条命令会在 `articles` 集合中的 `title` 和 `content` 字段上创建一个复合文本索引，意味着你可以对这两个字段进行全文搜索。

## **特性**

- **支持多个字段：** 一个文本索引可以跨多个字段。例如，如果文章有标题 (`title`) 和内容 (`content`)，你可以同时对这两个字段创建文本索引。
- **大小写不敏感：** MongoDB 会将文本字段的所有内容转化为小写，这意味着搜索是大小写不敏感的。
- **分词：** MongoDB 会自动根据空格和标点符号来分词，并为每个词建立索引。
- **支持停用词：** 一些常见的“无意义”词汇（如“the”、“a”、“and”等）会被忽略，除非你显式配置文本索引来使用这些词。
- **多语言支持：** MongoDB 提供了多种语言的分词器（如英语、中文、法语等）。

## **自定义停用词（针对非英文语言）**

如果你想为其他语言或特定需求自定义停用词集合，MongoDB 提供了选项来指定使用不同的分词器或自定义停用词列表。 MongoDB 通过 `default_language` 参数来指定语言，而通过 `language_override` 来让 MongoDB 根据特定字段的语言动态选择停用词。
#### **设置语言**

MongoDB 支持多种语言的分词器（包括中文、法语、德语等）。默认情况下，MongoDB 使用英语分词器，你可以通过在创建索引时显式指定语言来更改这一行为。

例如，设置为法语：

```javascript
db.collection.createIndex(
  { title: "text", content: "text" },
  { default_language: "fr" }
)
```

MongoDB 支持多种语言，以下是你可以使用的语言代码：

1. **阿拉伯语**: `ar`
2. **保加利亚语**: `bg`
3. **简体中文**: `zh`
4. **繁体中文**: `zh-Hant`
5. **捷克语**: `cs`
6. **丹麦语**: `da`
7. **荷兰语**: `nl`
8. **英语**: `en`
9. **爱沙尼亚语**: `et`
10. **芬兰语**: `fi`
11. **法语**: `fr`
12. **德语**: `de`
13. **希腊语**: `el`
14. **希伯来语**: `he`
15. **匈牙利语**: `hu`
16. **意大利语**: `it`
17. **日语**: `ja`
18. **韩语**: `ko`
19. **拉脱维亚语**: `lv`
20. **立陶宛语**: `lt`
21. **挪威语**: `no`
22. **波兰语**: `pl`
23. **葡萄牙语**: `pt`
24. **罗马尼亚语**: `ro`
25. **俄语**: `ru`
26. **西班牙语**: `es`
27. **瑞典语**: `sv`
28. **泰语**: `th`
29. **土耳其语**: `tr`
30. **乌克兰语**: `uk`

## **使用自定义语言（禁用默认的语言分词器）**

MongoDB 默认使用的分词器会自动应用默认语言的停用词集合（例如，英语）。虽然 MongoDB 不允许直接禁用停用词，但你可以选择一种分词器，它可能不会使用停用词，或者选择一个没有停用词的语言分词器。

**通过 `default_language` 参数控制语言**：你可以为文本索引指定不同的分词器，或指定一种没有停用词的语言。比如，使用 `none` 或者 `simple` 语言，这些语言的分词器不会应用停用词。

```javascript
db.collection.createIndex(
  { title: "text", content: "text" },
  { default_language: "none" }
);
```

这种设置指定了 `none` 语言分词器，这会避免使用任何语言的默认停用词。

**`none` 语言分词器**：`none` 是 MongoDB 提供的一种简单的分词器，它不会进行任何语言特定的分词，也不会有停用词过滤。

**`simple` 语言分词器**：`simple` 是另一个常见的分词器，它会去除一些最常见的停用词，但比默认的分词器过滤得更少。

```javascript
db.collection.createIndex(
  { title: "text", content: "text" },
  { default_language: "simple" }
);
```

## language_override

`language_override` 选项用于指定文档的某个字段来覆盖默认的语言设置。默认情况下，MongoDB 会根据 `default_language` 参数使用某种语言（通常是英语），来处理文本分析（如分词、停用词、词干提取等）。如果你的文档中包含多个语言的文本数据，或者文档中的某些字段需要特别的语言设置，你可以通过 `language_override` 来指定一个字段，覆盖默认语言。

```javascript
db.collection.createIndex(
   { field: "text" }, 
   { 
     default_language: "english", 
     language_override: "lang_field" 
   }
)
```

- `default_language`: 这是一个可选参数，用于设置默认语言（如果没有显式指定）。
- `language_override`: 这是你要指定的字段名称，MongoDB 会根据该字段的值来选择特定的语言。如果文档没有该字段，MongoDB 会使用 `default_language`。

```javascript
{
   "_id": 1,
   "title": "Hello World",
   "content": "This is a test document in English",
   "lang_field": "english"
}

{
   "_id": 2,
   "title": "Hola Mundo",
   "content": "Este es un documento de prueba en español",
   "lang_field": "spanish"
}
```

在这种情况下，你可以通过 `language_override` 来指定 `lang_field` 字段来覆盖每个文档的默认语言。

```javascript
db.collection.createIndex(
   { content: "text" },
   {
      default_language: "english",
      language_override: "lang_field"
   }
)
```

这样，`content` 字段将根据每个文档中的 `lang_field` 值来选择正确的语言：

- 对于第一篇文档，`lang_field` 的值是 `"english"`，所以会按照英语处理文本。
- 对于第二篇文档，`lang_field` 的值是 `"spanish"`，所以会按照西班牙语处理文本。
## 使用场景

创建了文本索引之后，你可以使用 `$text` 查询运算符来执行全文搜索。
### **简单查询**

假设你创建了一个包含标题 (`title`) 和内容 (`content`) 字段的文本索引，你可以使用 `$text` 运算符来搜索某个词汇或短语：

```javascript
db.articles.find({ $text: { $search: "MongoDB" } })
```

这会搜索 `title` 和 `content` 字段中包含 "MongoDB" 这个词的文档。
### **短语搜索**

你也可以搜索多个词汇组成的短语，只需要将词组放在双引号中：

```javascript
db.articles.find({ $text: { $search: "\"MongoDB Indexing\"" } })
```
### **排除某些词汇**

你可以通过在词汇前加上减号（`-`）来排除某些词汇。比如，我们要搜索包含 "MongoDB" 但不包含 "performance" 的文章：

```javascript
db.articles.find({ $text: { $search: "MongoDB -performance" } })
```

### **权重**

MongoDB 允许为文本索引中的字段分配权重，从而控制不同字段在搜索结果中的重要性。例如，如果你认为文章的 `title` 字段比 `content` 字段更重要，可以设置权重：

```javascript
db.articles.createIndex(
  { title: "text", content: "text" },
  { weights: { title: 10, content: 5 } }
)
```

在这个例子中，`title` 字段的权重是 `10`，而 `content` 字段的权重是 `5`。在搜索时，匹配到 `title` 的文档会被认为更相关，从而提高其排名。

### **文本搜索的额外操作**

MongoDB 可以返回匹配的文本字段的相关性得分。这个得分基于匹配度，越高的得分表示越相关的结果。你可以使用 `$meta` 运算符来获取这些得分。

```javascript
db.articles.find(
  { $text: { $search: "MongoDB" } },
  { score: { $meta: "textScore" } }
).sort({ score: { $meta: "textScore" } })
```

### **删除文本索引**

如果你不再需要某个文本索引，可以通过 `dropIndex` 删除它：

```javascript
db.articles.dropIndex("title_text_content_text")
```

这里的 `"title_text_content_text"` 是自动生成的索引名称，通常由字段名和索引类型组成。

## **示例**

假设你在开发一个博客系统，想要为文章的标题和内容添加全文搜索功能。

### 准备数据

```javascript
db.posts.insertMany([
  { title: "Introduction to MongoDB", content: "MongoDB is a NoSQL database." },
  { title: "MongoDB Indexing", content: "This article explains how to use MongoDB indexes." },
  { title: "MongoDB for Beginners", content: "Learn MongoDB basics in this tutorial." },
  { title: "Advanced MongoDB Features", content: "Explore advanced features like sharding and replication." }
])
```

###  **创建文本索引**

```javascript
db.posts.createIndex({ title: "text", content: "text" })
```

### **执行文本搜索**

**搜索 "MongoDB"：**

```javascript
db.posts.find({ $text: { $search: "MongoDB" } })
```

**搜索 "NoSQL" 并按相关性排序：**

```javascript
db.posts.find({ $text: { $search: "NoSQL" } })
  .sort({ score: { $meta: "textScore" } })
```

**排除某个词汇，搜索 "MongoDB" 但不包含 "tutorial"：**

```javascript
db.posts.find({ $text: { $search: "MongoDB -tutorial" } })
```

## **注意事项**

- **文本索引的大小：** 文本索引可能占用大量存储空间，尤其是在字段内容较长的情况下。要确保在创建文本索引时合理管理数据。
- **分词语言：** MongoDB 选择分词器时默认是英语。对于多语言环境，你可能需要调整分词器来支持其他语言。
- **性能：** 在数据量较大时，创建和维护文本索引可能影响写入性能。建议根据实际需求合理使用。

## 总结

MongoDB 的文本索引提供了一种高效的方式来执行全文搜索，可以对多个字段进行索引，支持复杂的查询操作如短语匹配、排除词汇、返回相关性得分等。合理利用文本索引，可以大大提升基于文本数据的搜索效率。










