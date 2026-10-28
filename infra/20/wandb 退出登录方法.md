# wandb 退出登录方法

## 🧑‍💻 User

wandb如何退出登录

---
## 😀 默认助手 (7)

## 结论先说

wandb **没有 `wandb logout` 这个命令**（很多人以为有）。我核对了官方 CLI 参考的完整命令列表（`agent / artifact / beta / clean / controller / disabled / docker / docker-run / enabled / init / job / leet / login / offline / online / pull / purge-cache / restore / server / status / sweep / sync / verify`），并直接在 `wandb` SDK 源码（`wandb/cli/cli.py`、`wandb/sdk/lib/wbauth`、`wandb/sdk/wandb_setup.py`）里检索 `logout`，**命中 0 次**。

所以"退出登录"的本质是：**删掉本机保存的 API Key**（或换成别的账号的 Key）。

---

## 凭据保存在哪里（按优先级）

`wandb.login()` 的查找顺序是：

| 顺序 | 位置 |
|---|---|
| 1 | 环境变量 `WANDB_API_KEY` |
| 2 | 设置文件里的 `api_key`：全局 `~/.config/wandb/settings`（或 `$WANDB_CONFIG_DIR`）、项目级 `./wandb/settings` |
| 3 | netrc：`~/.netrc`（Linux/macOS）、`~/​_netrc`（Windows），或 `$NETRC` 指定的路径 |
| 4 | 交互式输入 |

**这 4 处都得清掉，才算真正退出。**

---

## 方法 1：删除 netrc 里的凭据（最关键）

打开文件，删掉这一段：

```
machine api.wandb.ai

  login user

  password <你的-api-key>
```

- Linux / macOS：`~/.netrc`
- Windows：`C:\Users\<你>\_netrc`

macOS / Linux 上自动化删掉 wandb 条目（保留其他服务的凭据）：

```bash
python - <<'PY'
import os, pathlib
p = pathlib.Path(os.environ.get("NETRC") or pathlib.Path.home() / ".netrc")
blocks, cur = [], []
for ln in p.read_text().splitlines() + ["machine __END__"]:
    if ln.strip().lower().startswith("machine") and cur:
        blocks.append(cur); cur = [ln]
    else:
        cur.append(ln)
p.write_text("\n".join(l for b in blocks if not any("wandb" in x.lower() for x in b) for l in b) + "\n")
p.chmod(0o600)
print("done:", p)
PY
```

> ⚠️ 不要图省事直接 `rm ~/.netrc`——GitHub、其他 SaaS 的凭据也常存在这个文件里。注意文件权限必须是 `600`，否则会被忽略。

## 方法 2：清掉环境变量和设置文件

```bash
unset WANDB_API_KEY                    # 并从 ~/.bashrc / ~/.zshrc / Dockerfile / CI secrets 里删掉

grep -rn "api_key" ~/.config/wandb/settings ./wandb/settings 2>/dev/null
# 有输出就把对应行删掉（或直接删文件）
rm -f ~/.config/wandb/settings ./wandb/settings
```

## 方法 3：只是换账号 → `wandb login --relogin`

```bash
wandb login --relogin            # 强制重新登录，忽略已保存的 key
wandb login --relogin <NEW_KEY>  # 直接用新 key 覆盖
```

Python 中同理：

```python
import wandb
wandb.login(key="新的API_KEY", relogin=True)
```

## 方法 4：只想停止上传（不算退出登录）

```bash
wandb offline     # 当前项目目录 ./wandb/settings 写入 offline
wandb disabled    # 完全禁用
```

或临时设 `WANDB_MODE=offline` / `wandb.init(mode="offline")`。

## 网页端退出

浏览器里登录的是另一套会话：在 wandb.ai / W&B 控制台右上角头像菜单里选 **Log out** 即可（与本机命令行凭据互不影响）。

---

## 验证是否已退出

```bash
python -c "import wandb; print(wandb.login(prompt=False))"
```

输出 `False` 表示本机已无可用凭据，下次 `wandb.init()` 会提示重新登录。

<details>
<summary>执行过但没直接写进答案的细节</summary>

- 参考的 CLI 页面：`https://docs.coreweave.com/products/wandb/ref/cli/wandb-login`（W&B 文档现已并入 CoreWeave Docs）。
- 检索的源码：`wandb/cli/cli.py`（main 分支）、`wandb/sdk/lib/wbauth/__init__.py`、`wandb/sdk/wandb_setup.py`，`logout` 命中数均为 0。
- `wandb login` 成功提示原文：`Use wandb login --relogin to force relogin`——官方给出的"重新登录"方式就是它。
- 全局设置文件路径来自 `wandb/sdk/lib/settings_file.py` 的注释：`~/.config/wandb/settings`。
</details>

---
## 🧑‍💻 User

如果是root用户目录怎么删netrc（我是说普通用户，用了sudo东西在root里边怎么删）

---
## 🧑‍💻 User

如果是root用户目录怎么删netrc

---
## 😀 默认助手 (7)

## 先说结论

`/root` 的权限通常是 `drwx------ root root`，**普通用户根本无法读写里面的文件**，所以这件事只能靠 `sudo`（或 `su -`）来做——用 sudo 去删它是完全正常、也是唯一的做法。删完再回到你自己的账号跑 wandb 即可。

---

## 第 0 步：确认文件到底在哪

先弄清 sudo 时的 `HOME` 被设成了什么（你的情况应该就是 `/root`，这解释了 key 为什么跑进去）：

```bash
sudo printenv HOME
sudo find /root -maxdepth 3 \( -name .netrc -o -name _netrc \) 2>/dev/null
sudo ls -l /root/.netrc /root/.config/wandb/settings 2>/dev/null
```

> 别用 `sudo cat /root/.netrc`，那会把明文 API Key 打到屏幕上。要数条目用 `sudo grep -c machine /root/.netrc`。

如果 `sudo printenv HOME` 显示的不是 `/root`，说明 sudoers 保留了你的 HOME，那文件其实在你自己的家目录里，直接删就行，不需要 sudo。

## 第 1 步：删掉 netrc 里的 wandb 凭据

### 方案 A：只删 wandb 条目，保留 root 里其他凭据（推荐）

```bash
sudo python3 - <<'PY'
import pathlib
p = pathlib.Path("/root/.netrc")
if not p.exists():
    print("no /root/.netrc"); raise SystemExit
blocks, cur = [], []
for ln in p.read_text().splitlines() + ["machine __END__"]:
    if ln.strip().lower().startswith("machine") and cur:
        blocks.append(cur); cur = [ln]
    else:
        cur.append(ln)
keep = [b for b in blocks if not any("wandb" in x.lower() for x in b)]
drop = [b for b in blocks if any("wandb" in x.lower() for x in b)]
if not drop:
    print("no wandb entry"); raise SystemExit
p.write_text("\n".join(l for b in keep for l in b) + "\n")
p.chmod(0o600)
print("removed:", [b[0].split()[1] for b in drop if len(b[0].split()) > 1])
PY
```

不用 Python 的 awk 版本：

```bash
sudo awk '
  /^[[:space:]]*machine/ { if (buf != "" && tolower(buf) !~ /wandb/) printf "%s", buf; buf = $0 "\n"; next }
  { buf = buf $0 "\n" }
  END { if (tolower(buf) !~ /wandb/) printf "%s", buf }
' /root/.netrc | sudo tee /root/.netrc.new >/dev/null \
  && sudo mv /root/.netrc.new /root/.netrc \
  && sudo chmod 600 /root/.netrc
```

### 方案 B：直接删掉整个文件

确认过 root 的 netrc 里只有 wandb 的话，最省事：

```bash
sudo rm -f /root/.netrc
```

> 只想交互式操作也行：`sudo -i` 进入 root shell → `rm -f /root/.netrc` → `exit`。

## 第 2 步：顺带清掉 root 里的其他残留

sudo 跑过的 wandb 通常在 root 家目录留下不止一处：

```bash
# root 的全局设置文件里可能也有 api_key
sudo rm -f /root/.config/wandb/settings
# 或者只删 api_key 那一行
sudo sed -i '/^[[:space:]]*api_key/d' /root/.config/wandb/settings

# 如果你是 sudo wandb login <KEY> 这种形式，key 还留在 root 的 shell 历史里
sudo sed -i '/wandb login/d' /root/.bash_history

# sudo 跑训练脚本会在当前目录生成 root 属主的 ./wandb 目录
sudo chown -R "$USER":"$(id -gn)" ./wandb
```

## 第 3 步：验证

```bash
sudo test -f /root/.netrc && echo "还在" || echo "已删除"
sudo grep -c wandb /root/.netrc 2>/dev/null   # 0 或报文件不存在

# 如果你的 python 环境对 root 也可用，这个是终极验证
sudo python3 -c "import wandb; print(wandb.login(prompt=False))"   # False = 无凭据
```

**注意**：你自己账号下的验证和这个无关，普通用户跑 `wandb.login(prompt=False)` 只看你自己的 `~/.netrc`。

---

## 文件删了 ≠ 已退出：要不要吊销 Key

netrc 删掉只是让**这台机器上的 root** 不能用这个 key；key 本身在服务端依然有效。如果 root 家目录你并不信任（共享服务器、别人也能 sudo 等），去 wandb.ai → **Settings → API keys** 把它 **Revoke** 掉，这才是真正意义上的注销。

## 以后怎么避免

- **不要用 `sudo` 跑 `wandb login`**，用普通用户跑一次即可。
- 确实需要 root 跑任务时，用环境变量传 key，而不是让它去写 netrc：
  ```bash
  sudo --preserve-env=WANDB_API_KEY python train.py
  # 或指定放到你自己的配置目录
  sudo env WANDB_CONFIG_DIR=/home/你/.config/wandb python train.py
  ```
  （`sudo -H` 则会把 HOME 强制设成目标用户的家目录，按需选用；不同发行版 sudoers 行为有差异，先 `sudo printenv HOME` 确认。）
- 如果连 `sudo` 权限都没有，就只能 `su -`（需要 root 密码）或找管理员帮你删 `/root/.netrc`。
