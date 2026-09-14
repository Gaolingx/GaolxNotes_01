# ModelScope CLI上传指南

### 🧑‍💻 User

modelscope如何使用cli工具上传模型文件？

---

### 🤖 Assistant

# ModelScope CLI 上传模型文件的方法

ModelScope 的 CLI 命令是 `modelscope`（也可用别名 `ms`）。核心引擎由 `modelscope_hub` 提供。下面是完整的上传流程（命令语法已对照官方源码核实）。

## 1. 安装 CLI

```bash
pip install modelscope
# 或单独安装 hub 客户端
pip install modelscope_hub
```

安装后可用命令：`modelscope` / `ms`（也可用 `modelscope-hub` / `ms-hub`）。

## 2. 获取访问令牌（Access Token）

1. 登录魔搭社区 [modelscope.cn](https://modelscope.cn)
2. 进入 **个人中心 → 访问令牌（Access Token）**，生成一个 token（形如 `xxxxxxxx-xxxx-...`）。

## 3. 登录认证（三选一）

```bash
# 方式 1：交互式登录（会提示输入 token，输入时不可见）
modelscope login

# 方式 2：直接带 token 登录（token 会持久化保存到本地）
modelscope login --token <你的token>

# 方式 3：不登录，用环境变量 / 全局参数临时认证
export MODELSCOPE_API_TOKEN=<你的token>          # 环境变量
modelscope --token <你的token> upload ...        # 全局 --token 参数
```

验证登录身份：

```bash
modelscope whoami
```

## 4. 创建模型仓库（若尚未创建）

```bash
modelscope create <你的用户名/模型名> --repo-type model \
    --license Apache-2.0 \
    --description "模型描述" \
    --visibility public
```

例如：

```bash
modelscope create myname/my-first-model --repo-type model --license Apache-2.0
```

> 若仓库已存在可加 `--exist-ok` 避免报错。`--repo-type` 可选 `model` / `dataset` / `studio` / `skill`。

## 5. 上传模型文件 / 文件夹

基本语法：

```bash
modelscope upload <owner/仓库名> <本地路径> [仓库内路径] [选项]
```

### 上传单个文件

```bash
modelscope upload myname/my-first-model ./pytorch_model.bin
```

### 上传整个文件夹（最常见）

```bash
modelscope upload myname/my-first-model ./model_dir
```

上传目录时会递归上传 `model_dir` 下所有文件到仓库根目录，并自动生成 commit。

### 上传到数据集（dataset）

```bash
modelscope upload myname/my-dataset ./data_dir --repo-type dataset
```

### 上传到仓库内指定子目录

```bash
modelscope upload myname/my-first-model ./weights ./checkpoints
# 即：把本地 ./weights 上传到仓库的 ./checkpoints 目录下
```

## 常用参数速查

| 参数 | 说明 |
|------|------|
| `repo_id`（位置参数） | `owner/仓库名`，例如 `damo/nlp_corom_sentence-embedding_chinese-base` |
| `local_path`（位置参数） | 本地文件或文件夹路径；省略时自动推断（优先当前目录同名文件/目录） |
| `path_in_repo`（位置参数） | 目标在仓库内的路径，缺省为文件名本身或根目录 `/` |
| `--repo-type` / `--repo_type` | `model`（默认）或 `dataset` |
| `--revision` | 目标分支，默认 `master` |
| `--commit-message` | 本次提交的说明 |
| `--include GLOB...` | 仅上传匹配的文件（可重复，支持 glob） |
| `--exclude GLOB...` | 排除匹配的文件（可重复） |
| `--max-workers N` | 文件夹上传并发数 |
| `--use-cache` / `--no-cache` | 是否使用 `.ms_upload_cache` 断点续传（默认开启） |
| `--disable-tqdm` | 关闭进度条 |
| `--sync` | 同步模式：上传后删除远端多余文件 |

## 完整示例

```bash
# 1) 登录
modelscope login --token 91f8a7c1-xxxx-xxxx-xxxx-xxxxxxxxxxxx

# 2) 创建模型仓库
modelscope create myname/llm-demo --repo-type model --license Apache-2.0

# 3) 上传整个模型目录（含 config.json、权重等）
modelscope upload myname/llm-demo ./llm_demo \
    --commit-message "init model" \
    --include "*.json" "*.bin" "*.safetensors" \
    --exclude "*.pyc" "__pycache__/*"
```

## 补充说明

- **环境变量**：除了 `MODELSCOPE_API_TOKEN`，还可用 `MODELSCOPE_ENDPOINT` 指定自定义 API 端点（如内网镜像）。
- **断点续传**：文件夹上传默认使用 `.ms_upload_cache` 缓存，中断后重新执行同一命令即可续传；用 `--no-cache` 可关闭。
- **旧版语法兼容**：老版本 `modelscope` 的 `upload`/`create` 参数（如 `--model`、`--local_dir`）在新引擎下仍保留部分兼容别名，但官方推荐使用上文的 `owner/仓库名` 新语法。
- 若想纯 Python 方式上传，可用 `modelscope.hub.api.HubApi().upload_file(...)` / `upload_folder(...)` 或 `push_to_hub`，但 CLI 是最简便的方式。