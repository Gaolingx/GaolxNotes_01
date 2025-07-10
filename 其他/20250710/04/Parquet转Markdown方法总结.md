# Parquet转Markdown方法总结

### 🧑‍💻 User

parquet如何转markdown？

---

### 🤖 Assistant

将 Parquet 文件转换为 Markdown 表格通常需要借助编程工具，以下是两种常用方法：

---

### 方法 1：使用 Python（推荐）
#### 步骤：
1. **安装依赖库**：
   ```bash
   pip install pandas pyarrow tabulate
   ```

2. **转换代码**：
   ```python
   import pandas as pd

   # 读取 Parquet 文件
   df = pd.read_parquet("input.parquet")

   # 转换为 Markdown 表格并保存
   with open("output.md", "w") as f:
       f.write(df.to_markdown(index=False))  # index=False 表示不输出行索引
   ```

3. **执行脚本**：
   ```bash
   python convert.py
   ```

#### 特点：
✅ 支持大数据文件（自动分页处理）  
✅ 保留完整数据类型  
✅ 可定制列名/格式（通过 `df.rename()` 预处理）

---

### 方法 2：使用 DuckDB（命令行工具）
#### 步骤：
1. **安装 DuckDB**：
   [下载地址](https://duckdb.org/docs/installation/)

2. **执行 SQL 转换**：
   ```sql
   COPY (
     SELECT * FROM 'input.parquet'
   ) TO 'output.md' WITH (FORMAT MARKDOWN)
   ```

#### 特点：
✅ 无需 Python 环境  
✅ 极快处理速度（适合超大文件）  
✅ 支持复杂 SQL 预处理

---

### 转换效果示例：
原始 Parquet 数据：
| id | name    | value |
|----|---------|-------|
| 1  | Apple   | 7.2   |
| 2  | Banana  | 4.8   |

转换后的 Markdown：
```markdown
|   id | name   |   value |
|------|--------|---------|
|    1 | Apple  |     7.2 |
|    2 | Banana |     4.8 |
```

---

### 注意事项：
1. **大数据处理**：  
   若数据超 1000 行，建议分页处理（Python 中可分段写入）
2. **特殊字符**：  
   Markdown 表格中的 `|` 字符需转义为 `\|`
3. **空值处理**：  
   默认 `NaN` 会显示为空，可用 `df.fillna('N/A')` 替换

> 💡 **进阶技巧**：需要生成带样式的表格时，可在 Python 中使用 `prettytable` 库自定义表头格式。