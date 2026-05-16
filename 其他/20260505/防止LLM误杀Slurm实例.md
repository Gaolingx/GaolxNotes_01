# 防止LLM误杀Slurm实例

### 🧑‍💻 User

agent工具中，如何防止llm误操作把slurm实例全部杀掉？

---

### 🤖 Assistant

在构建与 Slurm 集群交互的 Agent 时，防止大语言模型（LLM）因为幻觉或误解用户意图而执行灾难性操作（例如 `scancel -u <user>` 杀掉所有任务）是非常核心的安全问题。

不要仅仅依赖 Prompt 来限制 LLM，因为 LLM 随时可能越狱或产生幻觉。最可靠的方法是采用**深度防御（Defense in Depth）**策略，在代码层、工具层和系统层进行硬性限制。

以下是防止 LLM 误杀 Slurm 实例的最佳实践：

### 1. 细粒度工具设计（最推荐）
**不要给 Agent 提供通用的“执行 Bash 命令”工具。** 如果你给了 Agent 一个 `execute_bash(command)` 工具，风险将不可控。你应该为 Slurm 操作设计专属的、受限的 API/函数。

强制 LLM 只能通过输入具体的 `job_id` 来取消任务，并且在代码层面进行强校验。

```python
def cancel_slurm_job(job_id: str) -> str:
    """
    取消指定的 Slurm 任务。
    参数:
        job_id: 必须是具体的数字 ID。
    """
    # 1. 强校验：确保传入的绝对是一个纯数字，防止注入 '-u username' 等参数
    if not job_id.isdigit():
        return "Error: job_id 必须是纯数字。"
  
    # 2. 执行安全的命令
    import subprocess
    try:
        # 使用列表形式传递参数，避免 shell 注入
        result = subprocess.run(['scancel', job_id], capture_output=True, text=True, check=True)
        return f"Successfully cancelled job {job_id}."
    except subprocess.CalledProcessError as e:
        return f"Failed to cancel job: {e.stderr}"
```

### 2. 引入 Human-in-the-loop (HITL) 人类确认机制
对于高危操作（如 `scancel`, `scontrol shutdown` 等），在工具实际执行前，强制拦截并要求人类用户授权。LangChain 和 AutoGen 等框架都支持这种模式。

```python
def safe_cancel_slurm_job(job_id: str) -> str:
    # 构造即将执行的动作
    action_desc = f"Agent 试图取消 Slurm 任务 ID: {job_id}"
    print(f"⚠️ 警告: {action_desc}")
  
    # 等待人类输入
    user_input = input("是否允许此操作？(y/N): ")
    if user_input.lower() != 'y':
        return "Action aborted by user."
  
    # 执行取消逻辑...
    return execute_cancel(job_id)
```

### 3. 命令正则黑名单/白名单拦截
如果你必须给 Agent 一个通用的命令行执行工具，你必须在 Python 代码层面对生成的 command 字符串进行正则拦截。

禁止任何带有批量语义的 `scancel` 参数（如 `-u`, `--user`, `-n`, `--name`, `*`）。

```python
import re

def execute_shell_command(command: str) -> str:
    # 黑名单正则：匹配 scancel 后面跟着 -u, --user, 或者不带具体数字ID的情况
    dangerous_patterns = [
        r"scancel\s+-u",
        r"scancel\s+--user",
        r"scancel\s+\*", 
        r"scancel\s*$" # 只有 scancel 没有参数
    ]
  
    for pattern in dangerous_patterns:
        if re.search(pattern, command):
            return "System Error: 执行被拒绝。Agent 不允许执行批量取消或危险的 scancel 命令。"
  
    # 正常执行...
```

<details>
<summary><b>点击展开：其他辅助性防御措施（Prompt 提示与系统权限）</b></summary>

虽然以下方法不能作为唯一的防线，但结合上述代码层面的限制，可以进一步降低风险。

**4. 系统层面的别名 (Alias) 或 Wrapper 脚本**
在 Agent 运行的宿主机环境中，将 `scancel` 命令替换为一个 Wrapper 脚本。即使 Agent 成功生成了恶意命令，系统底层也会拒绝。

创建一个伪造的 `scancel` 脚本放在 Agent 的 `PATH` 环境变量最前面：
```bash
#!/bin/bash
# Fake scancel wrapper

# 检查参数中是否包含 -u 或 --user
for arg in "$@"; do
  if [[ "$arg" == "-u" ]] || [[ "$arg" == "--user" ]]; then
    echo "Permission Denied: Batch cancellation is disabled for this agent."
    exit 1
  fi
done

# 如果安全，则调用真实的 scancel
/usr/bin/scancel "$@"
```

**5. System Prompt 的强力约束**
在 System Prompt 中明确规定 Slurm 操作的底线。这可以在源头上减少 Agent 产生错误意图的概率。

*System Prompt 示例:*
> You are an intelligent HPC assistant. When interacting with the Slurm scheduler, you must adhere to these strict rules:
> 1. NEVER cancel all jobs.
> 2. NEVER use `scancel -u` or `scancel --user`.
> 3. You may only cancel a job if the user explicitly provides the numeric `job_id`.
> 4. Always use the `cancel_slurm_job(job_id)` tool, and ensure the argument is a positive integer.

**6. 使用低权限的 Service Account**
不要让 Agent 运行在拥有所有任务权限的管理员账号（Root 或 SlurmAdmin）下。为 Agent 创建一个专属的普通 Linux 用户。虽然该用户仍然可以删掉“自己”的全部任务，但这限制了爆炸半径，避免了它删掉其他同事或集群的关键任务。
</details>

### 总结
最好的防御方案是 **方案 1（严格定义的 API 工具）** + **方案 2（高危操作需人工确认）**。永远不要信任 LLM 生成的 Raw Bash Command，必须通过中间件（Python 函数）将其转换为受限的、经过类型和安全校验的动作。