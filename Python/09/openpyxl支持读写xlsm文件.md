# openpyxl支持读写xlsm文件

### 🧑‍💻 User

openpyxl模块可以用来读写扩展名为xlsm的文件吗？

---

### 🤖 Assistant

**是的，`openpyxl` 模块可以用来读写扩展名为 `.xlsm` 的文件。**

`.xlsm` 是包含 VBA 宏的 Excel 文件格式。`openpyxl` 支持这种格式，但有一个**非常重要的前提**：你需要显式地告诉 `openpyxl` 在处理文件时保留宏代码，否则在保存时宏将会丢失。

### 如何读写 `.xlsm` 文件

在使用 `openpyxl.load_workbook()` 读取文件时，必须加上 `keep_vba=True` 参数。

以下是一个简单的代码示例：

```python
import openpyxl

# 1. 读取 .xlsm 文件，务必设置 keep_vba=True 以保留宏
file_path = 'example.xlsm'
wb = openpyxl.load_workbook(file_path, keep_vba=True)

# 2. 选择工作表并进行普通的读写操作
sheet = wb.active
print("A1单元格的原始值:", sheet['A1'].value)

# 修改数据
sheet['A1'] = '这是通过 openpyxl 修改的数据'

# 3. 保存文件
# 注意：保存时扩展名仍需为 .xlsm
save_path = 'example_modified.xlsm'
wb.save(save_path)

print("文件保存成功，宏已保留！")
```

<details>
<summary><b>⚠️ 重要的限制与注意事项（点击展开）</b></summary>

虽然 `openpyxl` 可以读写 `.xlsm` 文件，但它对 VBA 宏的支持仅限于**“携带”**（原样保留），具体限制如下：

1. **无法执行宏：** `openpyxl` 只是一个处理 XML 文件的 Python 库，它没有内置 Excel 引擎，因此**无法运行** `.xlsm` 文件中的任何 VBA 宏代码。
2. **无法读取或修改 VBA 代码：** 你不能使用 `openpyxl` 来查看、提取或修改文件内部的 VBA 脚本，它只是把包含宏的二进制部分当作一个黑盒原封不动地保存下来。
3. **另存为 `.xlsx` 的风险：** 如果你读取了一个开启了 `keep_vba=True` 的 `.xlsm` 文件，但最后使用 `wb.save('file.xlsx')` 将其保存为 `.xlsx` 格式，Excel 在打开这个新文件时通常会报错，因为 `.xlsx` 格式不允许包含 VBA 二进制数据。

**替代方案：**
如果你需要通过 Python **执行**或**修改** Excel 宏，你需要使用能够调用 Windows COM 接口的库（这意味着必须在安装了 Microsoft Excel 的 Windows 系统上运行），例如：
* `xlwings`
* `pywin32` (`win32com.client`)
</details>