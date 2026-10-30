# MCP与Nextflow区别

## 🧑‍💻 User

给我讲一下什么是mcp，如果开发生物信息的mcp有什么例子，比如说blast怎么做成mcp

---
## 😀 默认助手 (7)

我先查一下官方 MCP 注册表里现存的生物信息学服务器，这样给例子时能落到真实的项目上，而不是空谈。

注册表通过 npm 回退查询没有结果，我换个方式再查（包括直接查官方 registry API）。

官方 registry 的 "bio" 搜索基本被 "link-in-bio" 类服务淹没了，我改用几个更精确的关键词再筛一次。

下面是完整解答。先说结论：**在官方 MCP registry 里，我搜索 "blast" 只搜到 “blast radius / 命令影响范围评分” 这类同名项目，没有找到正经的生物信息学 BLAST server**——这说明 BLAST 作为 MCP 目前还是空白，你自己写反而很可能成为别人直接引用的实现。生信领域已经有的，基本是“数据库 REST API 封装”这一类，比如 UniProt 就至少有 5 个各自独立的实现（`cyanheads/uniprot-mcp-server`、`smaniches/uniprot-mcp`、`josefdc/uniprot-mcp` 等），另外还有 `biomcp`（PyPI，作者 Huajian Mao，定位就是 "MCP servers for bioinformatics"）。

# 一、MCP 到底是什么

一句话：**MCP（Model Context Protocol）是让 LLM 应用以统一方式“发现并调用”外部能力的开放协议**，2024 年底由 Anthropic 提出，现在已是事实标准（OpenAI、Google、各类 IDE 都支持）。

它要解决的不是“怎么调 API”，而是三个具体麻烦：

| 传统做法 | 有 MCP 之后 |
|---|---|
| 每个工具都硬编码进 prompt 或 function calling 列表 | server 自报 `tools/list`，带 JSON Schema，client 动态发现 |
| 参数描述靠人写文档 | 参数 schema + docstring 直接进模型上下文 |
| 权限、审批、审计各写一套 | 协议层统一 `elicitation`（中途向用户要确认）、`sampling`、progress、日志 |

**架构三层**：Host（Claude Desktop / Cherry Studio / IDE）→ Client（每个 server 一条连接）→ Server（你的程序）。**传输两种**：`stdio`（本地子进程，最常用）和 `streamable HTTP`（远程服务）。**原语三类**：`tools`（可执行动作）、`resources`（可读取的数据，如 `file:///...`、`blast://databases`）、`prompts`（预置模板）。

> 关键认知：**MCP 不等于“给 API 包一层 HTTP”**。它的价值在于让模型能自助发现工具、理解参数边界、并在长任务中交互。如果只是 1:1 映射 REST 端点，写 MCP 的收益有限；收益来自**为 LLM 重新设计的工具粒度**——这点在做 BLAST 时尤其致命。

# 二、为什么生物信息学特别适合 MCP

1. **工具生态极其碎片化**：BLAST、HMMER、SAMtools、BWA、GATK、Nextflow……几百个命令行工具，每个几十个参数。这正是 MCP schema 能发挥价值的地方。
2. **数据结构化程度高**：PDB、UniProt、Ensembl、NCBI 都有 REST/GraphQL，包装成本低，属于“容易摘的果子”。
3. **典型痛点就是“模型不知道该用哪个工具、参数怎么填”**：一个 BLAST 有 `-task`、`-word_size`、`-gapopen`、`-db` 十几个组合，MCP 的 schema 加默认值可以大幅减少幻觉参数。
4. **产物大、中间态多**：比对结果动辄几百 MB，必须“落盘 + 只回摘要”，这恰好是 MCP tool 返回值设计的核心约束。

# 三、现实中可参考的生物信息 MCP 形态

<details>
<summary><b>四类典型形态（点击展开）</b></summary>

**A. 数据库查询型（最主流）**——把 REST API 变成 3~6 个语义化 tool。以 UniProt 为例，典型工具集是：`search_proteins(query)` / `get_protein(accession)` / `map_ids(ids, from, to)` / `get_sequence(accession)`。这类 server 一天能写完，价值在于让模型用自然语言做结构化的 Swiss-Prot 查询。

**B. 领域知识型**——`biomcp` 这类，把 PubMed、ClinicalTrials、变异数据库、基因注释打包成一套面向“科研问答”的工具集，重点在**跨库联动**（一个变异 → 相关文献 → 相关试验）。

**C. 计算执行型**——把本地/集群上的命令行工具暴露出去。这是 BLAST 所属的类别，也是**难度最高的一类**，难点全在长任务和产物管理，见下文。

**D. 交互分析与可视化型**——结构预测、序列比对可视化、单细胞结果浏览，依赖 `resources` 把图片/文件回传。

</details>

# 四、把 BLAST 做成 MCP：4 个必须先想清楚的设计决策

这是本题的核心。**90% 的人第一版会写成 `run_blast(sequence, db, ...) -> str` 并在 tool 里同步等 5 分钟结果，然后被客户端超时打断——这是最大的坑。**

### 决策 1：绝对不要同步阻塞

一次 `blastn` 对 `nt` 库跑 2 kb 序列，几分钟到几十分钟。MCP client 侧（尤其 IDE 和桌面端）普遍有 60s 级超时，而且**工具调用期间模型是“卡住”的**（该轮次无法与用户对话）。

正确形态是**拆成状态机**：

```
submit_blast() → {job_id, status:"running"}   ← 立即返回
get_blast_status(job_id) → {status, elapsed}
get_blast_results(job_id, top_n, min_pident) → 结构化摘要 + 落盘路径
```

长任务期间的进度用 `ctx.report_progress()` / `ctx.info()` 推给客户端；想让用户确认“这个库要跑 20 分钟，确定吗”就用 `ctx.elicit()`。

> 规范里已有 `tasks`（异步任务）原语，但客户端支持参差。**务实的做法就是自己实现 submit/poll**，兼容性最好。

### 决策 2：工具粒度要“面向意图”，不要照搬命令行

反模式：`run_blast(query, db, outfmt, evalue, word_size, gapopen, gapextend, max_target_seqs, num_threads, dust, soft_masking, ...)` —— 14 个参数，模型必填错。

正模式：**按任务分工具 + 参数收敛到 5 个以内 + 其余给安全默认值**。

```
blast_nucleotide(seq, database="core_nt", task="megablast", max_hits=50)
blast_protein(seq, database="swissprot", max_hits=50)
list_blast_databases()        # 让模型先自己查有什么库
describe_database(name)       # 库的物种/条数/版本 —— 可复现性的关键
```

`describe_database` 这个工具很值钱：模型的“知识”里没有你本地库的版本，不告诉它，它就会在报告里编造数据库版本。

### 决策 3：输出必须“摘要 + 指针”，绝不回灌原始对齐

`-outfmt 6` 跑 50 条命中轻松几十 KB，直接塞回 context 会**挤爆上下文窗口，并让后续推理质量下降**。

方案：
- BLAST 写 TSV 到**磁盘**；
- tool 只返回 top N（默认 10~20）+ 过滤（pident ≥ x、evalue ≤ y）+ 标题截断；
- 同时返回 `full_result_path`，让模型需要细节时再读（配合 `resources` 或文件读取工具，做**渐进式披露**）；
- 附带轻量富化字段（物种、identity 分箱），减少模型二次查库。

### 决策 4：安全边界必须是硬约束

BLAST 参数里 `-db` 是**路径**，如果拼命令字符串就等于开了 RCE。要求：

- `create_subprocess_exec(*cmd)`（数组传参，**永不 `shell=True`**）
- 数据库走**白名单 + 存在性校验**，不接受任意路径
- 查询长度/大小上限、`-num_threads` 上限、任务数上限（防并发把机器打爆）
- 超时后**显式 kill** 子进程，否则僵尸进程会累积

# 五、最小可用实现（Python + FastMCP）

```python
# blast_mcp/server.py
import asyncio, csv, os, uuid
from datetime import datetime, timezone
from pathlib import Path
from mcp.server.fastmcp import FastMCP, Context

mcp = FastMCP("blast-mcp")

DB_DIR      = Path(os.environ.get("BLAST_DB_DIR", "/data/blastdb"))
WORKDIR     = Path(os.environ.get("BLAST_WORKDIR", "/tmp/blast-mcp"))
ALLOWED_DBS = set(filter(None, os.environ.get("BLAST_ALLOWED_DBS", "").split(",")))
MAX_THREADS = int(os.environ.get("BLAST_MAX_THREADS", "8"))
MAX_QLEN    = int(os.environ.get("BLAST_MAX_QUERY_LEN", "100000"))
WORKDIR.mkdir(parents=True, exist_ok=True)

JOBS: dict[str, dict] = {}

OUTFMT = ("6 qseqid sseqid pident length mismatch gapopen qstart qend "
          "sstart send evalue bitscore staxids sscinames stitle")

def _db_path(name: str) -> Path:
    # 双保险：白名单 + 必须真实存在，且不允许包含路径分隔符
    if name not in ALLOWED_DBS or "/" in name or ".." in name:
        raise ValueError(f"database {name!r} not allowed. allowed={sorted(ALLOWED_DBS)}")
    p = DB_DIR / name
    if not any(p.with_suffix(s).exists() for s in (".nin", ".pin", ".nsq", ".psq", ".pal")):
        raise ValueError(f"database {name!r} not found under {DB_DIR}")
    return p

@mcp.tool()
async def list_blast_databases() -> list[str]:
    """列出本机可用的 BLAST 数据库名称。调用其他 blast 工具前建议先查一次。"""
    return sorted(ALLOWED_DBS)

@mcp.tool()
async def describe_database(database: str) -> dict:
    """返回数据库的物种、序列条数、总长度和构建时间，用于判断适用性与结果可复现性。"""
    db = _db_path(database)
    proc = await asyncio.create_subprocess_exec(
        "blastdbcmd", "-db", str(db), "-info",
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
    out, _ = await proc.communicate()
    return {"database": database, "blastdbcmd_info": out.decode(errors="replace").strip()}

@mcp.tool()
async def submit_blast(
    sequence: str,
    database: str = "core_nt",
    program: str = "blastn",
    task: str | None = "megablast",
    max_hits: int = 50,
    evalue: float = 1e-5,
) -> dict:
    """提交一个 BLAST 任务，立即返回 job_id（不阻塞）。
    program: blastn | blastp | blastx | tblastn。序列可以是 FASTA 或纯序列。
    结果请用 get_blast_results 获取。"""
    if program not in {"blastn", "blastp", "blastx", "tblastn"}:
        raise ValueError("program must be blastn/blastp/blastx/tblastn")
    if len(sequence) > MAX_QLEN:
        raise ValueError(f"query too long ({len(sequence)} > {MAX_QLEN})")

    job_id = uuid.uuid4().hex[:12]
    qfile  = WORKDIR / f"{job_id}.fa"
    qfile.write_text(sequence if sequence.startswith(">") else f">query\n{sequence}\n")
    out = WORKDIR / f"{job_id}.tsv"
    log = WORKDIR / f"{job_id}.log"

    cmd = [program, "-query", str(qfile), "-db", str(_db_path(database)),
           "-out", str(out), "-outfmt", OUTFMT,
           "-max_target_seqs", str(max_hits), "-evalue", str(evalue),
           "-num_threads", str(MAX_THREADS)]
    if task:
        cmd += ["-task", task]

    JOBS[job_id] = {"job_id": job_id, "status": "running", "database": database,
                    "program": program, "out": str(out), "log": str(log),
                    "submitted_at": datetime.now(timezone.utc).isoformat()}
    asyncio.create_task(_run_bg(job_id, cmd, log))
    return {"job_id": job_id, "status": "running",
            "next": "poll get_blast_status(job_id) until status != 'running'"}

async def _run_bg(job_id: str, cmd: list[str], log: Path, timeout: int = 3600) -> None:
    with log.open("w") as errf:
        proc = await asyncio.create_subprocess_exec(*cmd, stderr=errf)
        try:
            rc = await asyncio.wait_for(proc.wait(), timeout=timeout)
            JOBS[job_id].update(status="done" if rc == 0 else "failed", returncode=rc)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            JOBS[job_id].update(status="timeout", returncode=-1)

@mcp.tool()
async def get_blast_status(job_id: str) -> dict:
    """查询任务状态：running / done / failed / timeout。"""
    job = JOBS.get(job_id)
    if not job:
        raise ValueError(f"unknown job_id {job_id!r}")
    return {k: v for k, v in job.items() if k != "cmd"}

@mcp.tool()
async def get_blast_results(
    job_id: str, ctx: Context, top_n: int = 15, min_pident: float = 0.0,
) -> dict:
    """任务完成后读取命中结果。只返回 top_n 条摘要，完整比对见 full_result_path。"""
    job = JOBS.get(job_id) or {}
    if job.get("status") == "running":
        return {"status": "running", "hint": "still running, poll get_blast_status"}
    if job.get("status") != "done":
        return {"status": job.get("status"), "log": _read(job.get("log", ""), 40)}

    hits, total = [], 0
    with open(job["out"], newline="") as fh:
        for row in csv.reader(fh, delimiter="\t"):
            if len(row) < 15:
                continue
            total += 1
            pident = float(row[2])
            if pident < min_pident:
                continue
            if len(hits) >= top_n:
                continue
            hits.append({
                "subject_id": row[1], "identity_pct": round(pident, 1),
                "align_len": int(row[3]), "evalue": float(row[10]),
                "bitscore": float(row[11]), "accession": row[1].split("|")[1]
                    if row[1].count("|") >= 2 else None,
                "taxid": row[12].split(";")[0] or None,
                "species": row[13] or None,
                "title": (row[14][:110] + "…") if len(row[14]) > 110 else row[14],
            })
    await ctx.info(f"{total} raw hits parsed, returning {len(hits)}")
    return {"job_id": job_id, "database": job.get("database"),
            "total_hits": total, "returned": len(hits), "hits": hits,
            "full_result_path": job["out"],
            "note": "如需全部命中，请读取 full_result_path（TSV, outfmt 6）"}

def _read(path: str, tail: int) -> str:
    try:
        return "\n".join(Path(path).read_text(errors="replace").splitlines()[-tail:])
    except OSError:
        return ""

@mcp.tool()
async def get_blast_log(job_id: str, tail: int = 40) -> str:
    """查看任务日志尾部，任务失败时优先看这里。"""
    return _read(JOBS[job_id]["log"], tail)

if __name__ == "__main__":
    mcp.run()   # 默认 stdio
```

客户端配置：

```json
{
  "mcpServers": {
    "blast": {
      "command": "python",
      "args": ["-m", "blast_mcp.server"],
      "env": {
        "BLAST_DB_DIR": "/data/blastdb",
        "BLAST_ALLOWED_DBS": "core_nt,swissprot,16S_ribosomal_RNA",
        "BLAST_MAX_THREADS": "8"
      }
    }
  }
}
```

写完立刻能验证：在支持 MCP 的客户端里问“本地有哪些 BLAST 库，帮我确认 swissprot 的版本和物种范围”——如果它能先调 `list_blast_databases` 再调 `describe_database`，说明工具描述写对了。

<details>
<summary><b>变体：不装本地 BLAST，直接调 NCBI 远程 URL API（点击展开）</b></summary>

适合没有本地库、或想跑 `core_nt` 这种几百 GB 库的场景。流程是 **PUT 拿 RID → 轮询 → 取结果**，同样要拆成 submit/poll：

```python
import httpx, asyncio

NCBI = "https://blast.ncbi.nlm.nih.gov/Blast.cgi"

async def submit_remote(sequence: str, program="blastn", database="core_nt", megablast=True):
    params = {"CMD": "Put", "PROGRAM": program, "DATABASE": database,
              "QUERY": sequence, "MEGABLAST": "on" if megablast else "off",
              "HITLIST_SIZE": 50}
    async with httpx.AsyncClient(timeout=60) as c:
        r = await c.post(NCBI, data=params)
    for line in r.text.splitlines():
        if "RID = " in line:
            return line.split("RID = ")[1].strip()
    raise RuntimeError("no RID returned; NCBI may be rate-limiting")

async def poll_remote(rid: str, interval=15, max_wait=1800):
    waited = 0
    async with httpx.AsyncClient(timeout=60) as c:
        while waited < max_wait:
            r = await c.get(NCBI, params={"CMD": "Get", "RID": rid,
                                          "FORMAT_OBJECT": "SearchInfo"})
            if "Status=WAITING" not in r.text:
                if "Status=FAILED" in r.text:
                    return {"status": "failed"}
                return {"status": "ready"}
            await asyncio.sleep(interval); waited += interval
    return {"status": "timeout"}

async def fetch_remote(rid: str, fmt="Tabular"):
    async with httpx.AsyncClient(timeout=120) as c:
        r = await c.get(NCBI, params={"CMD": "Get", "RID": rid, "FORMAT_TYPE": fmt})
    return r.text   # Tabular 仍是文本，需自己切列
```

三个必须注意的点：**NCBI 对无 API key 的请求限速 3 req/s**（轮询间隔别低于 10s）；**RID 有时效（约 36 h 后失效）**；**远程结果格式不如本地 `-outfmt` 规整**，解析要容错。生产环境建议还是本地库 + 定时 `update_blastdb.pl`。
</details>

# 六、踩坑清单（做之前先看）

<details>
<summary><b>工程与设计细节（点击展开）</b></summary>

**必踩的**
- **同步阻塞**：见决策 1。第二常见的错误是“加个 `timeout` 参数让用户自己等”。
- **上下文爆炸**：返回 `outfmt 0`（人类可读比对）或全量 TSV，一次几十万 token，模型直接失能。永远 top-N + 落盘。
- **任务状态丢失**：上面代码里 `JOBS` 在内存中，server 重启就没了。生产上把状态写成 `WORKDIR/{job_id}/state.json`，启动时扫描目录恢复。
- **子进程泄漏**：超时只 `kill()` 不 `await proc.wait()` 会留僵尸；另外 `blastn` 会派生线程，必要时用 `start_new_session=True` + `os.killpg`。

**做了就加分的**
- **工具描述就是 prompt**：`submit_blast` 的 docstring 会被送进模型。写明“不阻塞、请轮询”、“database 请先用 list_blast_databases 确认”，模型行为会明显变好。参数用 `Field(description=...)` 或 Annotated 补语义。
- **`readOnlyHint` 标注**：`list_blast_databases`、`describe_database` 是纯读，标注后客户端可以免确认。
- **`resources` 暴露库清单**：`@mcp.resource("blast://databases")` 让库列表作为可读数据出现，模型不需要专门调工具。
- **`prompts` 固化流程**：比如 `@mcp.prompt("annotate_contig")` 预置“先 megablast 找近缘 → 再 blastx 找 ORF → 汇总物种”的提示，把领域经验交给用户。
- **可复现性**：结果里带上 `blast_version`、库的 `blastdbcmd -info` 快照、完整命令行。生信领域这点是刚需，也是审稿/复核时唯一能救命的东西。
- **对齐你真正的工作流**：如果实验室实际跑的是 Nextflow/Snakemake 流水线，那 MCP 更适合包**流水线提交**（`submit_pipeline(samplesheet, params)`）而不是单个 `blastn`；BLAST 只是其中一步。

**别做的事**
- 别把 28 个 `blastn` 参数 1:1 暴露出去。
- 别用 `shell=True` 拼命令行（`-db` 是路径，等于后门）。
- 别一个 server 塞 40 个工具——工具超过 ~20 个，模型的选择准确率会明显下降，按领域拆 server。
</details>

# 七、如果想发布出去

1. **打包**：PyPI 上 `pip install` + `uvx your-mcp`，或 Docker 镜像（生信依赖重，Docker 明显更友好：把 `blast+` 和库一起烤进镜像，用户 `docker run` 就能用）。
2. **双传输**：`stdio` 给本地桌面端，`streamable HTTP` 给远程/多用户。
3. **注册**：向官方 registry 提交 `server.json`（含 `repository`、`packages`、`transport`、环境变量描述），之后别人就能搜到——目前生信 server 在这个 registry 里还非常稀疏，先发优势明显。
4. **定位建议**：**不要做“BLAST MCP”，要做“序列相似性搜索 MCP”**——下面挂 `blastn`、`diamond`、`minimap2`、`hmmscan`，对外统一成 `search(seq, mode="nucleotide|protein|long-read")`。这不仅更符合 MCP 的“面向意图”哲学，也避免模型纠结该选哪个命令行工具。

---

需要的话，我可以接着给你：① 上面代码的可运行完整工程（含 `pyproject.toml`、Dockerfile、blastdb 初始化脚本）；② 或者把它换成 TypeScript SDK 版本；③ 或者按同一套“submit/poll/摘要”模式，示范一个 Nextflow 流水线 MCP。

---
## 🧑‍💻 User

那mcp其实就不是开发一个新的工具，是把原本工具里面的各种功能再进行封装，是这样吗

---
## 😀 默认助手 (7)

这个判断**有一半是对的，但那一半可能让你掉进最大的坑**。

## 对的部分：MCP 本身不创造任何能力

MCP server 里几乎不存在“新算法”。BLAST 还是那个 `blastn`，UniProt 还是那个 REST API。**MCP 是接口层，不是能力层**——它把“模型知道该调什么、怎么调、结果怎么理解”这件事标准化了。

这一点你已经抓住了。所以如果有人说“我做了个 BLAST MCP，实现了序列比对”，严格讲他实现的只是**比对工具的接入方式**。

## 不那么对的部分：“封装”不等于“包装”

这里有个关键区别，正好对应上一轮那个 BLAST 例子：

| | 原始 `blastn` | 我给你的 MCP server |
|---|---|---|
| 调用方式 | 阻塞，跑完才返回 | `submit` 立刻返回 `job_id` |
| 状态 | 无（进程活着就是跑着） | 显式 `status` 状态机 |
| 输出 | 写文件 / 几十 MB TSV | top-N 摘要 + 落盘指针 |
| 参数 | 20+ 个 flag | 5 个语义化参数 |
| 库选择 | `-db /任意/路径` | 白名单校验 |

**`submit/poll` 这个状态机在 `blastn` 里根本不存在**，`top-N + 指针` 这种输出策略也不存在。这些是**为 LLM 重新设计的东西**，不是“把已有功能暴露出去”。所以：

> MCP 的产物不是“功能”，是**接口契约**。而设计一个好的契约，是真正的工作量所在。

## 三层封装，价值密度完全不同

<details>
<summary><b>展开：L1 / L2 / L3</b></summary>

**L1 代理层（thin proxy）**——REST/CLI 端点 1:1 映射成 tool。
写起来最快，`search_proteins` 对应 `/uniprotkb/search`。UniProt 那 5 个实现基本都在这层。价值主要来自“统一接入”，不是设计。

**L2 重设计层（adapter with semantics）**——拆分、聚合、裁剪、富化、加安全边界。
BLAST 那个 server 在这层。**绝大多数优质 MCP server 都停在这里**，因为 LLM 的约束（上下文有限、会超时、会幻觉参数）在这里被真正照顾到。

**L3 编排层（orchestration）**——一个 tool = 一条多工具流水线。
比如 `annotate_contig(seq)`，内部依次跑 `blastn` → 找 ORF → `blastx` → 汇总物种 → 出表。**这一层是真正的新能力**，因为原本没有任何一条命令能干这事。领域知识（该用什么阈值、先跑哪个库）被固化进了工具。

L3 才是“开发新工具”，L1 只是“接线”。
</details>

## 最好的类比不是 API，是 LSP

如果 MCP 只是“API 的包装”，那它确实没啥意思。但它更像 **LSP（Language Server Protocol）**：

LSP 没有发明编译器，它只是把编译器能力标准化成协议。听起来也是“封装”。**但正因为标准化，才催生了“语言服务器”这个独立物种**——VS Code 不用为每种语言改编辑器，语言作者也不用为每个 IDE 写插件。十年下来，语言服务器能做的事远超“把编译器包一下”（诊断、重构、语义高亮、内联提示），因为**有了协议，就有了专门为交互场景设计接口的空间**。

MCP 在同一个位置。协议标准化之后，才会有人去做 L2/L3 那些“原始工具里没有、但交互场景需要”的接口设计。

## 但你的怀疑指向了一个真问题

如果只做 L1，很多时候**不如直接给 agent 一个 shell**。这个观点有道理，Anthropic 自己就讨论过 “code execution with MCP”——让模型写代码去调工具，往往比逐个 tool call 更省上下文、更灵活。

所以一个 L1 server 的合理性必须靠这些撑住，否则不如给命令行：

- **权限可控**：只放开白名单操作，不是整台机器的 shell
- **结构化输出**：模型不用解析人类可读文本
- **非技术用户可用**：桌面端用户不会敲 `blastn -task megablast ...`
- **跨客户端复用**：写一次，Claude/cherry studio/IDE 都能用
- **可审计**：每次调用都是显式的、有 schema 记录的

反过来，**如果你只是把 `ls` 包了个 tool，那确实没有意义**——那正是 shell 存在的理由。

## 一条实用判据

<details>
<summary><b>什么时候“封装”就够，什么时候必须重设计（点击展开）</b></summary>

**L1 够用**：已有干净的 API/CLI、单次调用秒级返回、输出小于几 KB、无状态、无破坏性。
→ 典型：UniProt 查询、PubMed 检索、ID 映射、密码子翻译。

**必须 L2**：任务超过客户端超时、输出会撑爆上下文、参数组合爆炸、有文件/路径风险、需要用户中途确认。
→ 典型：BLAST、GATK、任何测序比对。

**值得 L3**：领域内有“老师傅才知道的正确步骤顺序”，且这个顺序反复被问。
→ 典型：变异注释、单细胞标准流程、从 reads 到物种鉴定。

还有一个反直觉的：**粒度不是越细越好**。把 `run_blast` 拆成 `open_file`/`set_param`/`run`/`close` 这种“原子操作”是灾难——工具超过 20 个，模型的选择准确率就开始掉。**按用户意图切，不按实现步骤切。**
</details>

## 一句话收束

你的描述在“MCP 不发明算法”上完全正确；但把它的产出说成“封装已有功能”会低估工作量——**MCP 真正在做的，是把工具的“能力”翻译成 LLM 能安全、可靠、低成本使用的“接口契约”**。翻译得好不好，决定了这个 server 是能用的工具还是没人装的 demo。

要不要我拿你手头某个具体工具（比如某个组内脚本或 HMMER/minimap2）走一遍，直接判断它该落在 L1 还是 L2/L3、工具怎么切？

---
## 🧑‍💻 User

那mcp和nextflow pipeline的区别是什么，我怎么感觉有点像

---
## 😀 默认助手 (7)

你感觉像是有道理的——**两者在“形状”上确实同构**：都是“不实现算法、只做胶水”的一层，都靠一份声明式描述（`main.nf` + `nextflow.config` / `server.json` + JSON Schema）配一个运行时引擎。但如果只看到这层相似，会做出很糟的架构。

**一句话结论：它们不在同一层，是正交的、互补的。MCP 是给模型的接口层，Nextflow 是给集群的执行层。**

## 最本质的区别：方向相反

这是我最想让你记住的一点：

| | Nextflow | MCP |
|---|---|---|
| 不确定性 | **收敛**它 | **开放**它 |
| 决策时机 | 编译期（DAG 先定死） | 运行期（模型当场决定） |
| 谁做决定 | 写 pipeline 的人 | LLM + 用户对话 |
| 消费者 | 调度器 / 执行引擎 | 模型 / agent |

Nextflow 的整个价值主张是：**把人类那些靠脑子的、不确定的步骤，提前固化成一条可重复的、确定性的 DAG**。你写 `main.nf` 的那一刻，就承诺了“这批数据永远这么跑”。

MCP 干的是相反的事：**把确定性的接口交给一个非确定性的模型，让它实时决定这次该怎么组合**。同一个 server，今天被调用 3 次、明天被调用 11 次，顺序还不一样——这是特性，不是 bug。

所以：**Nextflow 消灭不确定性，MCP 利用不确定性。**

## 逐项对比

| 维度 | Nextflow | MCP |
|---|---|---|
| 解决什么 | 依赖、并行、容错、断点续跑、资源调度 | 工具发现、参数契约、交互、权限 |
| 状态 | 一次提交的 DAG，跑完即止（`-resume` 靠缓存） | 常驻进程，有会话、可中途问用户 |
| 并行 | 核心抽象是 channel 扇出，天然处理 N 个样本 | 一次请求一个结果，N 个样本就得调 N 次 |
| 缓存 | `-resume` 基于内容哈希，可靠 | **几乎无法缓存**——输入是自然语言，不可哈希 |
| 失败 | `errorStrategy`、retry、`work/` 哈希定位 | 抛错给模型，模型自己决定重试/换路/问你 |
| 可复现 | 容器 + 版本锁，接近 bit-level | 本身不保证，得靠自己把版本写进返回值 |
| 数据流 | 一等公民（channel） | request/response，大产物靠落盘+指针 |
| 粒度 | process = 一个数据单元 | tool = 一个用户意图 |

**最实用的一条**：缓存和并行的能力必须下沉到 Nextflow 层。**你没法在 MCP 层做缓存**（“帮我比对这条序列”两次措辞不同，但其实是一回事），所以任何批量化需求都应该交给 pipeline，MCP 只负责“按下开始键”。

## 正确的组合方式：分层，不是二选一

```
用户自然语言
      ↓
┌─────────────────────────────┐
│ MCP server   ← 决策层/交互层 │  选参数、校验 samplesheet、解释日志、
│                              │  出错时问用户、把结果翻译成人话
└─────────────────────────────┘
      ↓ 生成 samplesheet + params.json，提交
┌─────────────────────────────┐
│ Nextflow     ← 执行层        │  并行、容器、重试、-resume、资源
└─────────────────────────────┘
      ↓
  容器化工具 (blastn / samtools / nf-core modules)
```

一个很具体的例子：**nf-core pipeline 有几百个参数，这是 MCP 的主要应用场景**。用户说“我有 12 个双端 RNA-seq 样本，人源，想看看差异表达”，MCP 负责判断该用 `nf-core/rnaseq`、该传什么 `--genome`、samplesheet 格式对不对、内存该给多少，然后调用 `nextflow run`；剩下的并行和重试由 Nextflow 管。

```python
@mcp.tool()
async def submit_nfcore_rnaseq(samplesheet_csv: str, genome: str = "GRCh38",
                               aligner: str = "star_salmon") -> dict:
    """提交 nf-core/rnaseq。会先校验 samplesheet 列名与文件路径，再启动 Nextflow。
    立即返回 run_id，用 get_run_status 轮询。"""
    # 1. 校验（这就是 MCP 的价值：在昂贵的计算之前拦住错误）
    # 2. 写 samplesheet / params.yaml
    # 3. asyncio.create_subprocess_exec("nextflow", "run", "nf-core/rnaseq", ...)
    # 4. return {"run_id": ..., "status": "submitted"}
```

注意第 1 步——**“在花 6 小时机时之前发现 samplesheet 少了一列”**，这是 MCP 在生信里最被低估的价值，Nextflow 自己做不到（它只会在 process 启动时才崩）。

## 什么时候用哪个：三个判断

<details>
<summary><b>展开：决策清单 + 两个反模式</b></summary>

**用 Nextflow**
- 样本数 > 3，或需要并行
- 要跑几小时以上，需要断点续跑
- 结果要交给别人复现（审稿、交付、合规）
- 同样的流程会重复很多次

**用 MCP**
- 一次性探索性分析（“这条序列像什么”）
- 需要人在环中做判断（“这两个命中哪个更可信？”）
- 要接的是**数据库/API** 而不是重计算（查 UniProt、查文献）——这类根本没有 pipeline 的必要
- 用户是非技术背景，需要自然语言入口

**反模式 1：用 MCP 编排批处理**
让模型对 100 个样本逐个调 `submit_blast`。慢、贵、易错、失败没法续跑、无法缓存。**这时候你应该让 MCP 生成一个 samplesheet 然后提交 Nextflow**，一个 tool call 解决。

**反模式 2：为一次分析写 pipeline**
为了跑一条序列的 BLAST，去写 `main.nf` + `nextflow.config` + Dockerfile。半天过去了。探索阶段应该 MCP + 直接调工具，**等流程稳定了再沉淀成 pipeline**。这个沉淀路径本身就是好的工程节奏：`手动调 → MCP 固化 → Nextflow 固化`。
</details>

## 另一个容易混的点

你可能会想：“那 MCP 里写个 `run_blast`，Nextflow 里写个 `process BLAST`，不也一样吗？”

不一样。Nextflow 的 `process` 是**无状态的、单数据单元的、可由引擎重排和并行的**；MCP 的 tool 是**有会话上下文的、接受模糊输入的、可以被用户打断的**。前者要幂等，后者要能解释“为什么失败”。

顺带说：理论上 Nextflow process 里也能去调 MCP server（HTTP），但基本没意义——DAG 都定死了，再让模型实时决策等于自己跟自己打架。

---

**收束成一句**：Nextflow 回答“**怎么可靠地把这 1000 个样本跑完**”，MCP 回答“**模型怎么知道你手上有哪些工具、怎么替你把它按下去**”。前者是编译器后端，后者是驱动层。

如果你愿意，我可以把你现在实际在跑的那条流程（哪些工具、几个样本、给谁用）拉到这个框架里，直接判断哪一段该固化进 Nextflow、哪一段该包成 MCP。

---
## 🧑‍💻 User

我觉得 MCP本质就是某种CLI的包装形式吧（仅限tools） 多数情况CLI比MCP好多了
MCP的主要优势我觉得还是资源
但是MCP的资源很强大（可惜的是 很多client都不会正确实现MCP resources）
MCP tools其实远不如CLI（如果agent有bash的话）
其实MCP的用法很多
对于tools包装CLI 其实仅限于写代码 很多其他场景MCP tools是很有用的（虽然我觉得MCP最好的部分还是resources）
不过 现在各大client和provider对工具热重载的支持还不够 tools的最大问题就是热重载

比如我自己开发的py based miniagent 我有十几个function 我把这个写成mcp就还是让ai去调这些function 把不同的生物分析流程用mcp串起来 可本身他自己用cli不就够了
怎么说呢 要不是不想让agent碰bash谁做MCP这种
MCP tools在代码场景下就很难比CLI好
MCP也没有原生progressive disclosure
我的意思很多时候我们其实不需要专用harness 尤其是像生物信息分析这种 根本就是纯cli的东西 为什么要脱裤子放屁 直接让ai去读代码写cli不比mcp强吗
给agent一个bash环境和python啥都搞定了

总之，MCP的价值更多在于restricted access
像代码这种 不需要考虑太多安全和隐私的 就不太需要MCP
因为 CLI需要一个外部环境 也就是fs
但是MCP不需要 因为它可以直接host在agent client里
然后 MCP不适合这种有开放性的任务
MCP比较适合构建一些特定业务流程 比如客服

最后 随着模型能力越来越强 可以慢慢放下很多先验了
可能mcp和harness某种程度上是上了缰绳 限制的同时更准确了
但也会丢失一些ambitious results

---
## 😀 默认助手 (7)

大方向上我同意你八成，但有一个 reframe 能让剩下的分歧都变清楚：**问题在于 MCP tools 是“枚举式约束”，CLI 是“组合式约束”，两者约束的强度差不多，可扩展性差了一个数量级**。而“安全 vs 自由”这个二分法并不成立——bash 也是约束，只是它是一套组合能力，而不是 15 个固定函数。

## 你说对的地方（不重复，只补刀）

**1. 自己的 agent 包自己的函数，MCP 是纯开销。** 这一点没有争议。MCP 的价值只在**跨边界**时兑现：跨组织、跨进程、跨语言、跨权限域、跨发布节奏。你既是 provider 又是 consumer，边界为零，于是你付了 JSON-RPC 序列化 + schema 维护 + 进程管理的钱，换来一个“看起来更规范”的调用。**对独立开发者，MCP 的 ROI 是负的**，这不是你用得不对。

**2. 生信就是纯 CLI + 文件系统**，这一点没法反驳。CLI 还额外给了你几件 MCP 至今没有的能力：

| CLI 原生有 | MCP tools 没有 |
|---|---|
| `PATH` + 子命令，天然命名空间 | 所有 server 的 tool 摊平成一个列表，只能靠 `blast_submit` 这种前缀 hack |
| `stderr` + exit code + 可管道 | 一个错误字符串，没法 grep、没法试 |
| 可以读 `man`/`--help` 现学 | 靠 schema，schema 写不全就废 |
| 可以组合成新命令 | 只能按 provider 预设的组合调用 |

**3. 没有原生 progressive disclosure。** 严格说有半个：`tools/list` 支持 `cursor` 分页，还有 `notifications/tools/list_changed`。但**语义层面**确实没有——没有 tool 分组、没有检索、没有“相关性到了再加载”。有意思的是生态里的解法是“做一个 tool，用 `action` 参数分子命令”——**这等于把 CLI 重新发明了一遍**。你的直觉在这一点上是被验证的。

**4. 热重载确实是真痛点**，而且比你说的更麻烦：它不只是 client 不实现 `list_changed`，更是**状态问题**。server 是常驻进程，重启就丢掉内存里的 job registry（就是上一轮 BLAST 那个 `JOBS` 字典）。所以好的 server 必须把状态外置到磁盘——热重载做不好，一半责任在协议，一半在“有状态 server 本身”这个设计。

## 我要纠正的两点

**（a）MCP 不是沙箱，它给的比你想象的少。**

“MCP 不需要 fs、可以 host 在 client 里”——这个安全感是假的。stdio server 就是一个**以你的用户权限运行、默认无沙箱**的子进程，它对文件系统的访问权和你的 bash 一样多，甚至更不可见（用户看不见它在干嘛）。

MCP 真正给的是：**可枚举的动作空间 + 最小权限白名单 + 可审计的调用记录**。这是“允许列表”，不是“隔离”。真要隔离，得靠 bwrap/nsjail/容器/seccomp，跟 MCP 完全正交。而且**一个恶意 MCP server 比 shell 更危险**，因为用户信任它、不会看它干什么。

**（b）resources 之所以烂，不全怪 client——是设计层面缺了东西。**

你说“resources 是最有价值的部分”，我同意方向，但它失败有个结构性原因：**resources 没有 schema**。tools 有 JSON Schema 可以让模型推理；resources 只有 uri + mimeType，模型在“读它”之前对它一无所知——不知道多大、什么结构、读了值不值。没有成本模型，就没法做 progressive disclosure，client 自然懒得渲染。

更深一层：**resources 的 `subscribe` 是 push/订阅模型，而 agent 循环是 pull/轮次的**。resources 在架构上比运行时超前了一代。这就是为什么生态最后收敛成“干脆做成一个 `read_xxx` tool”。**是 designs 的问题导致了 client 的冷漠**，不完全是 client 懒。

## 关于“缰绳”那段，我想岔开一个轴

你说“harness 是缰绳，限制的同时更准确，但会丢失 ambitious results”。我基本同意，但我觉得**这里混了两个轴**：

- 轴一：**能力补偿**（模型不够强，所以要拆步骤）→ 这个随模型变强确实该拆
- 轴二：**认识论基础设施**（结果要可复现、可审计、可追责）→ 这个**不随模型变强而消失**

在科研场景里，第二类约束不是“怕模型算错”，而是**输出的用途决定的**。“记录 BLAST 库版本”这条 requirement，模型智商 200 也需要，因为审稿人需要。所以 reins 不会消失，是**从程序性先验迁移到不变式先验**——从“第 1 步做什么”变成“无论你怎么做，这几件事必须为真”。

而且你的“丢失 ambitious results”我觉得方向可能反了：**给 bash 的模型比给 15 个函数的模型做更 ambitious 的事**。因为 shell 是 affordance（能力开放、可组合），function list 是 enumeration（能力封闭）。所以不是“缰绳 vs 自由”，是“**错误的缰绳 vs 更宽的缰绳**”。

## 那 MCP 到底还有没有位置

有，但客户画像要换。**MCP 真正的客户不是 agent 开发者，是 tool provider**——他们想“让所有 agent 都能调用我的服务”，于是付了封装成本换分发。这就是为什么 registry 里全是 SaaS 厂商，而不是科研工具作者。

在生信里，唯一真正站得住的场景是：**湿实验的人用 Claude Desktop，而你不能给他 bash。** 这些人数量在快速增长，且他们**必须**被限制在 15 个安全动作里——不会用 `-num_threads`，也不该能 `rm -rf`。对这批人，MCP 的成本是合理的门票。对你自己，直接给 bash 就行。

**给你的 miniagent 的判据**：

- 只是不想让它碰 fs → 用 **sandbox（bwrap/容器）+ 一个 `bio` CLI 的 15 个子命令**，别写 MCP
- 想让同事在 Cherry Studio / Claude Desktop 里用 → 写 MCP，这钱花得值
- 想要跨语言/跨机器调用 → 写 MCP（或干脆 HTTP + OpenAPI）
- 只是自己迭代 → 函数 + schema 就够了，别再抽象一层

顺带一个判断：**你那个“把不同生信流程用 MCP 串起来”的想法，如果流程本身是确定性的，那它其实是 Nextflow 的活；如果是探索性的，那它其实是“让模型写 bash 脚本”的活。MCP 夹在中间，两头不占。**

最后一点——我觉得最有意思的是：**你这套推论在“模型能力变强”这个方向上会自我强化**。模型越会写代码，能用 CLI 的场景就越多，MCP 的合理边界就越窄，最后收缩到“权限边界 + 分发通道”。这个收缩是健康的，只是别把中间的过渡期当成终态。
