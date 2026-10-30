# 云平台建站服务比较 Cloudflare AWS GCP

## 🧑‍💻 User

cloudflare aws gcp有哪些建站相关云服务

---
## 😀 默认助手 (7)

## 一句话定位差异

| 平台 | 建站定位 | 典型使用场景 |
|---|---|---|
| **Cloudflare** | 边缘网络优先，免费额度给得很充足，**“托管与加速”是核心**，计算属于次要能力 | 静态站点、JAMstack 站点、给已有站点加 CDN、WAF、DNS |
| **AWS** | 服务覆盖完整，从裸机实例到 SaaS 的全栈服务都有，**粒度最细，配置也最繁琐** | 中大型动态站、企业级架构、需要精细控制成本与合规 |
| **GCP** | 容器与数据层最强，Firebase 负责前端侧，**开发者体验较好** | 容器化应用、移动端、与 Firebase 集成的 Web 应用 |

---

## 按建站环节对比

### 1. 域名 / DNS

| 环节 | Cloudflare | AWS | GCP |
|---|---|---|---|
| 域名注册 | **Cloudflare Registrar**：以成本价出售，不加价 | Route 53 Domains | Cloud Domains API 仍在用，但面向消费级市场的注册业务已转给 Squarespace，目前不适合作为域名注册商的首选 |
| DNS 托管 | DNS，**免费套餐就提供完整的权威 DNS 服务，还带有 DNSSEC 与代理功能** | Route 53 Hosted Zone，按查询量收费 | Cloud DNS，按区域与查询量收费 |
| 泛播加速解析 | 免费套餐即包含 Anycast 加速 | Route 53 使用 Anycast，需要付费 | Cloud DNS 使用 Anycast，需要付费 |

> 只把 DNS 托管到 Cloudflare，让站点以橙云代理模式接入，是性价比最高的起步方式，现有的站点不必改动架构。

### 2. 静态托管 / CDN / 边缘

| 能力 | Cloudflare | AWS | GCP |
|---|---|---|---|
| 静态站点托管 | **Pages**，免费，从 Git 仓库构建、提供预览环境、支持自定义域 | **Amplify Hosting**；或 S3 静态网站 + CloudFront | **Firebase Hosting**（体验最好）；或 Cloud Storage + 负载均衡 |
| CDN | 全球 Anycast 边缘网络，**免费套餐就包含完整的 CDN** | CloudFront，按流量计费 | Cloud CDN / Media CDN，需配合负载均衡器 |
| 边缘函数 | **Workers**，有一个 V8 isolate 运行时，免费额度大 | CloudFront Functions、Lambda@Edge | 无真正的边缘 JS 运行时，只能靠 Cloud Run + 全球均衡近似实现 |
| SSR / 全栈框架 | Pages Functions、Workers（支持 Next.js/Nuxt/SvelteKit 适配） | Amplify Hosting 支持 SSR，另有 App Runner、Lambda | Cloud Run（容器化 SSR，最灵活）、Firebase App Hosting |
| 图片优化 | Images、Image Resizing、Polish | 无原生支持，需 Lambda@Edge + Sharp 自行搭建 | 无原生支持，需 Cloud Run + Sharp 自行搭建 |
| 视频 | **Stream**，提供转码、播放器和按观看计费 | MediaConvert + IVS / CloudFront | Transcoder API + Cloud Storage |

### 3. 计算（跑动态站）

| Cloudflare | AWS | GCP |
|---|---|---|
| Workers（Serverless 边缘） | Lambda（Serverless） | Cloud Functions / Cloud Run functions |
| Containers（2024 年起提供） | App Runner（容器托管，可固定运行实例） | **Cloud Run**：可无服务器运行、按请求伸缩、也可固定运行实例，**体验最好** |
| —（Workers 中不能运行完整 Node 进程） | ECS / EKS / Fargate | GKE / GKE Autopilot |
| — | EC2、**Lightsail**（有 WordPress、LAMP 等预装模板，接近 VPS 的使用方式） | Compute Engine（含一键部署 WordPress、LAMP 的 Marketplace 镜像）、App Engine |

> **WordPress 之类的传统站点**：可以使用 AWS Lightsail 或 GCP Compute Engine + Marketplace（也可使用 Cloudflare Tunnel 把自托管的站点藏到内网）；但从成本和抗攻击能力来看，更推荐放在 Cloudflare 后面。

### 4. 存储 / 数据库

| 类型 | Cloudflare | AWS | GCP |
|---|---|---|---|
| 对象存储 | **R2**（S3 兼容，**出口流量免费**） | S3 | Cloud Storage |
| KV / 键值 | Workers KV | DynamoDB | Firestore / Bigtable |
| SQL | **D1**（SQLite 形态） | RDS / Aurora（MySQL/PG/…) | Cloud SQL / AlloyDB / Spanner |
| 缓存 | Durable Objects、Cache API | ElastiCache | Memorystore |
| 队列 | Queues | SQS / SNS / EventBridge | Pub/Sub |
| 向量库 | Vectorize | OpenSearch / Bedrock KB | Vertex AI Vector Search |

### 5. SSL / 安全 / 认证

| 能力 | Cloudflare | AWS | GCP |
|---|---|---|---|
| 免费 SSL | **Universal SSL：免费证书覆盖根域与泛域名** | ACM，免费证书，仅对 AWS 上的服务有效 | Google 管理的 SSL 证书，免费 |
| WAF | **WAF 有免费规则集；完整版靠 Pro/Business 套餐付费** | AWS WAF，按规则和请求计费 | Cloud Armor |
| DDoS | 默认开启，且免费 | Shield Standard 默认开启、免费 | Cloud Armor 默认的 DDoS 防护 |
| 人机验证 | **Turnstile：可替代 reCAPTCHA，免费且注重隐私** | 无原生服务，可用 WAF Bot Control | reCAPTCHA Enterprise |
| 用户认证 | Cloudflare Access / Zero Trust | Cognito | **Identity Platform / Firebase Auth** |
| 密钥管理 | Secrets Store | Secrets Manager / KMS | Secret Manager / Cloud KMS |
| 内网穿透 | **Cloudflare Tunnel：可以免费安全地向公网暴露本地或内网服务** | 无原生等价能力，需自行用 VPN 或 ALB 实现 | Cloud IAP |

### 6. 邮件 / 表单 / 分析

| 环节 | Cloudflare | AWS | GCP |
|---|---|---|---|
| 收信转发 | **Email Routing，免费，可转发到任意邮箱** | SES 收信，配置较麻烦 | 无原生服务 |
| 事务邮件发送 | 无原生服务，可用 MailChannels | **SES**，便宜且成熟 | 无原生服务，常用 SendGrid / Mailgun |
| 企业邮箱 | 无 | WorkMail | Google Workspace |
| 表单后端 | Workers + D1 | API Gateway + Lambda | Cloud Run / Firebase |
| 访问统计 | **Web Analytics**，免费、不依赖 Cookie | CloudWatch + 自建 | 同左，另有 Looker Studio |
| 第三方脚本管理 | Zaraz | — | — |

### 7. 构建 / 部署 / 运维

| | Cloudflare | AWS | GCP |
|---|---|---|---|
| 内置 Git 构建 | **Pages、Workers Builds**（免费） | Amplify Hosting | Firebase App Hosting / Cloud Build |
| CI/CD | Workers Builds | CodePipeline / CodeBuild / CodeDeploy | Cloud Build / Cloud Deploy |
| 日志 | Logpush | CloudWatch / CloudTrail | Cloud Logging |
| IaC | Wrangler、Terraform | CDK / CloudFormation / Terraform | Deployment Manager / Terraform |

---

## 三家“一站式建站路线”对比

<details>
<summary><strong>最省心的组合（推荐给个人、小团队、博客、文档站）</strong></summary>

```
Cloudflare Registrar（域名）
  → Cloudflare DNS（橙云代理）
  → Cloudflare Pages（前端托管 + 免费 CI）
  → Workers / Pages Functions（API 层）
  → D1 / R2（数据 + 静态资源）
  → Turnstile（表单防刷）
```
**成本：域名 + 0 元**，含全球 CDN、DDoS 防护、SSL 证书。

</details>

<details>
<summary><strong>AWS 典型组合（企业、需要精细控制）</strong></summary>

```
Route 53（域名 + DNS）
  → S3（静态资源）
  → CloudFront（CDN + WAF + ACM 证书）
  → Amplify Hosting（前端 SSR / CI）
  → ECS Fargate 或 App Runner（后端）
  → RDS / Aurora（数据库）
  → SES（邮件）+ Cognito（认证）
  → CodePipeline + CloudWatch（运维）
```
特点：功能全面，但**账单最容易失控**（NAT Gateway、跨区流量、CloudFront 回源费用）。

</details>

<details>
<summary><strong>GCP 典型组合（容器化、Firebase 生态）</strong></summary>

```
Cloud DNS + Cloud Domains（域名）
  → Firebase Hosting（前端 + SSR + 自动 SSL）
  → Cloud Run（后端容器，可零实例休眠）
  → Cloud SQL 或 Firestore
  → Cloud Storage + Cloud CDN（大文件）
  → Firebase Auth + Identity Platform
  → Cloud Build（CI/CD）
```
特点：`Cloud Run` 是三家 Serverless 容器里最好用的；GCP **没有原生事务邮件服务**，这是明显短板。

</details>

---

## 选型建议（按建站类型）

| 你的站点类型 | 首选 | 理由 |
|---|---|---|
| 纯静态站、博客、文档站 | **Cloudflare Pages + R2 / D1** | 免费额度足够，构建快，自带 CDN |
| Next.js / Nuxt 全栈 | Cloudflare Workers，或 Vercel / Firebase App Hosting | Workers 的 SSR 适配已经比较成熟 |
| WordPress | Lightsail / GCE Marketplace 自建 + Cloudflare 前置 | 要降低攻击面；注意插件带来的漏洞 |
| 企业官网 + 会员系统 | AWS（Cognito + RDS）或 GCP（Identity Platform + Cloud SQL） | 认证与合规体系更完整 |
| 电商 | AWS（成熟支付、库存生态）或 GCP + Shopify 等 SaaS | Cloudflare 没有原生电商能力 |
| 小程序、移动端的后端 | **GCP Firebase** 或 Cloudflare Workers + D1 | 集成度高，SDK 现成 |

---

## 容易遇到的问题

1. **Cloudflare 免费 CDN 只对 HTTP/HTTPS 流量生效**，Tcp/UDP 需要走 Spectrum（付费）；WebSocket 支持。
2. **S3 静态网站托管不能直接用 HTTPS**，必须挂 CloudFront + ACM（ACM 证书必须放在 `us-east-1`）。
3. **Cloud Storage 静态网站托管需要**配置负载均衡 + Cloud CDN 才能拿到自定义域和 HTTPS。
4. **AWS 免费套餐是 12 个月限时**，GCP 的 always-free 是长期有效（e2-micro、Cloud Run 免费额度），Cloudflare Pages/Workers 的免费额度最稳。
5. **Cloudflare 要求域名托管使用它自己的 DNS** 才能使用代理和 WAF；只把 CDN 接入作为补充能力使用也可以实现，但功能和体验会受限。
6. **GCP 的 Cloud Domains 已不再是面向消费级用户的主要域名注册商**（Google Domains 业务 2023 年转给 Squarespace），新建站点建议在 Cloudflare 或 Route 53 注册，再改 NS 指向。
7. **R2 的免费出口流量只针对 R2 对外传输流量**，通过 Cloudflare 代理回源仍是免费的，这是它相对 S3 的最大优势。

需要我针对某个具体场景（比如“个人博客 + 评论 + 图片”或“公司官网 + 后台”）出一份具体的选型组合和成本估算吗？
