# Linuxzip压缩

## 🧑‍💻 User

linux zip压缩目录命令

---
## 🤖 Assistant

# Linux 目录 zip 压缩命令

## 基本用法

```bash
zip -r 压缩包名.zip 要压缩的目录
```

**示例**：把 `mydir` 目录压缩成 `mydir.zip`

```bash
zip -r mydir.zip mydir
```

> `-r` 表示递归（recursive），压缩目录**必须**加此参数，否则只会打包空壳。

---

## 常用参数

| 参数 | 含义 |
|------|------|
| `-r` | 递归压缩子目录（目录必用） |
| `-q` | 静默模式，不显示过程 |
| `-9` | 最高压缩率（更慢） |
| `-1` | 最快压缩（压缩率低） |
| `-e` | 加密，会提示输入密码 |
| `-x` | 排除指定文件 |
| `-m` | 压缩后删除源文件 |
| `-v` | 显示详细信息 |

---

## 实用例子

**压缩并排除某些文件**
```bash
zip -r project.zip project/ -x "*.git*" -x "*.log"
```

**压缩时设置密码**
```bash
zip -r -e secret.zip mydir/
```

**静默 + 最高压缩率**
```bash
zip -r -q -9 backup.zip /path/to/dir
```

**只压缩目录内容（不含自身这一层）**
```bash
cd mydir && zip -r ../mydir.zip .
```

---

## 相关命令

**查看压缩包内容**
```bash
unzip -l mydir.zip
```

**解压**
```bash
unzip mydir.zip              # 解压到当前目录
unzip mydir.zip -d /target   # 解压到指定目录
```

**测试压缩包完整性**
```bash
unzip -t mydir.zip
```

---

## 注意事项

- 如果提示 `bash: zip: command not found`，需先安装：
```bash
# Debian/Ubuntu
sudo apt install zip unzip
# CentOS/RHEL
sudo yum install zip unzip
```

- `zip` 对 **中文文件名** 兼容性一般，跨平台传输时建议用 `tar.gz`：
```bash
tar -czvf mydir.tar.gz mydir/
```
- `zip` 不支持的场景（如保留 Linux 权限、软链接）建议用 `tar`。

如果你有具体的使用场景（比如定时备份、排除规则），可以告诉我，我帮你写出对应命令。
