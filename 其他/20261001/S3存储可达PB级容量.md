# S3存储可达PB级容量

## 🧑‍💻 User

aws s3能塞pb级的存储空间吗？

---
## 😀 默认助手 (7)

**能，而且远远不止 PB。** S3 在总容量和对象数量上**没有任何硬上限**——AWS 官方措辞是 "virtually unlimited"（事实上的无限）。已有多家客户在 S3 上跑 EB 级（$1\,\text{EB} = 1024\,\text{PB}$）数据。真正的约束不在"能不能装下"，而在**单对象大小、请求速率、成本和列举/管理方式**这几处。

## 关键限制一览

| 维度 | 限制 | 备注 |
|---|---|---|
| 存储桶总容量 | 无上限 | 单个 bucket 无限大 |
| 对象数量 | 无上限 | 单 bucket 可放百亿+对象 |
| **单个对象大小** | **最大 5 TiB** | 这是唯一真正的容量级硬限制 |
| 单次 PUT | 最大 5 GiB | 更大必须走 Multipart Upload |
| Multipart 分片 | 最多 10,000 片，每片 5 MiB–5 GiB | 要存满 5 TiB，分片需 ≥ 512 MiB |
| 每账户 bucket 数 | 默认 100，可申请提升 | 软限制 |
| 前缀级吞吐 | 3,500 PUT/COPY/POST/DELETE 或 5,500 GET/HEAD 每秒**每 prefix** | prefix 数量无上限，可线性扩展 |
| Bucket Policy 大小 | 20 KB | PB 级场景下这个很容易撞墙 |

## PB 级真正会踩的坑

1. **5 TiB 单对象上限**：一个 PB 级数据集不可能塞进一个对象。要么拆成大量对象，要么在应用层做分片（sharding）。若确实需要超大逻辑文件，可用 S3 Batch Operations 或客户端拼接。
2. **Policy 20 KB 限制**：如果你打算用"IAM policy 里逐 prefix 授权"的方式管理 PB 数据，很快就会撑爆。正确做法是**按角色 + 前缀约定授权**，而不是堆 policy 条目。
3. **列举（LIST）不可行**：LIST 每次只返回 1,000 个 key。10 亿对象的 bucket 靠 LIST 遍历是灾难。PB 级必须上 **S3 Inventory**（支持 Parquet/ORC，可被 Athena/Spark 直接查询）+ **S3 Storage Lens**（29+ 项指标）。
4. **流量费用 > 存储费用**：这是最容易被忽略的一点。跨区/NAT 出流量按 $0.09/GB 量级计费，把 1 PB 传出一次约 **$94,000**。生产环境务必用 **VPC Endpoint（Gateway Endpoint 免费）**，避免走 NAT Gateway。

<details>
<summary>PB 级存储成本估算（us-east-1 量级，1 PB = 1,048,576 GB，仅为存储费）</summary>

| 存储类 | 单价 ($/GB-月) | 1 PB / 月 | 取回特性 |
|---|---|---|---|
| S3 Standard | $0.023（分层，大客户可降至 $0.021） | ≈ $22,600–24,100 | 毫秒级 |
| S3 Intelligent-Tiering | 自动分层 + 监控费 | 视访问模式，通常降 30–70% | 毫秒级 |
| S3 Standard-IA | $0.0125 | ≈ $13,100 | 毫秒级，有取回费 |
| S3 One Zone-IA | $0.01 | ≈ $10,500 | 单 AZ，耐久性降为 11 个 9 → 实际略低 |
| S3 Glacier Instant Retrieval | $0.004 | ≈ $4,200 | 毫秒级 |
| S3 Glacier Flexible Retrieval | $0.0036 | ≈ $3,800 | 分钟–小时 |
| S3 Glacier Deep Archive | $0.00099 | ≈ $1,040 | 12 小时左右 |

</details>

## 实用建议

- **性能型场景**（AI/HPC 训练读 PB 数据）：用 **S3 Express One Zone**（目录桶，个位数毫秒延迟）或挂载 **FSx for Lustre** 直接读 S3。
- **冷数据归档**：Deep Archive 能把 1 PB 压到约 $1,000/月，但取回按 GB + 请求计费，且需提前规划。
- **合规/分层**：生命周期策略自动从 Standard → IA → Glacier → Deep Archive，能省掉大头成本。
- **耐久性**：S3 Standard 提供 11 个 9 的持久性，数据跨 ≥3 个 AZ 冗余存储。

一句话：**PB 级对 S3 是常规量级，没有任何容量障碍；需要设计的是对象大小切分、prefix 并发、Inventory 式管理与出流成本。**

如果你能告诉我具体场景（数据量、对象平均大小、读写模式、是否要训练/AI 用途），我可以给一个更具体的架构和成本方案。
