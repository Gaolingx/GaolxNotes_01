# PowerShell临时添加npm路径到PATH

## 🧑‍💻 User

powershell添加npm路径设置临时环境变量

---
## 😀 默认助手 (7)

# PowerShell 临时添加 npm 路径到 PATH

## 最简做法：当前会话临时生效

```powershell
# 追加 npm 全局 bin 目录（默认在 %APPDATA%\npm）
$env:Path = "$env:APPDATA\npm;" + $env:Path
```

`$env:` 前缀的变量只影响**当前 PowerShell 窗口及其子进程**，关闭窗口即失效，不会写注册表，这正是“临时环境变量”。

## 完整步骤

### 1. 先确认 npm 的真实路径

```powershell
Get-Command npm, node | Select-Object Name, Source
npm config get prefix
```

常见结果：

| 安装方式 | npm 全局 bin 路径 |
|---|---|
| 官方 Node 安装包 | `C:\Program Files\nodejs`（npm 本身）+ `%APPDATA%\npm`（全局包 shim） |
| nvm-windows | `C:\Users\<你>\AppData\Roaming\nvm\<版本>` + `C:\Program Files\nodejs`（符号链接） |
| 自定义 prefix | `npm config get prefix` 的输出 |

### 2. 临时追加（二选一）

```powershell
# 推荐：用 $env:APPDATA 拼，避免写死用户名
$env:Path = "$env:APPDATA\npm;" + $env:Path

# 或写死绝对路径
$env:Path += ";C:\Program Files\nodejs;C:\Users\me\AppData\Roaming\npm"
```

> 注意：`$env:Path += ";..."` 和 `$env:Path = "...;" + $env:Path` 都可以，但**必须带分号**。漏掉分号会把两段路径拼成一个不存在的目录。

### 3. 验证

```powershell
$env:Path -split ';' | Select-String 'npm|nodejs'
npm -v
node -v
```

### 4. 只给某个子进程用的“临时”

如果不想污染当前会话，只想让一条命令带上路径：

```powershell
$env:Path = "$env:APPDATA\npm;$env:Path"; npm run build
# 或者用 .NET 的 Process 作用域（同样只影响本进程）
[Environment]::SetEnvironmentVariable('Path', "$env:APPDATA\npm;$env:Path", 'Process')
```

<details>
<summary>对比：持久化设置（非临时，供参考）</summary>

```powershell
# 用户级，永久写入注册表，需重开终端生效
[Environment]::SetEnvironmentVariable('Path', [Environment]::GetEnvironmentVariable('Path','User') + ";$env:APPDATA\npm", 'User')

# 或者用 setx（注意：setx 有 1024 字符截断风险，且不展开 %VAR%）
setx PATH "%PATH%;%APPDATA%\npm"
```

新手最常见的坑：直接 `setx PATH "%PATH%;..."` 在 PATH 很长时会**截断并破坏原有 PATH**。持久化优先用 `[Environment]::SetEnvironmentVariable(..., 'User')`。

</details>

## 几个容易踩的坑

- **变量名大小写/拼写**：PowerShell 里是 `$env:Path`（也可 `$env:PATH`，Windows 不区分大小写），但别写成 `$Env:PATH` 之外的 `$PATH`——`$PATH` 是普通 PowerShell 变量，跟环境变量无关。
- **引号**：路径含空格时，`$env:Path` 字符串里**不要**给单条路径加双引号，`C:\Program Files\nodejs` 直接写即可；加了引号反而可能让某些工具解析失败。
- **脚本里重复执行会不断变长**：写进 `$PROFILE` 前建议先判断，避免 PATH 无限膨胀：

  ```powershell
  $npmBin = "$env:APPDATA\npm"
  if ($env:Path -notlike "*$npmBin*") { $env:Path = "$npmBin;$env:Path" }
  ```

- **`npm` 能找到但全局包命令找不到**：那是 `%APPDATA%\npm` 没进 PATH；反之 `npm -v` 报“无法识别”才是 `nodejs` 目录没进 PATH，两者是不同的目录。
- **执行策略**：如果改 `$PROFILE` 后不生效，先看 `Get-ExecutionPolicy`，可能是 `Restricted` 导致 profile 没加载。
