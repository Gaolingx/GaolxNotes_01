# Nginx与Tomcat区别及Node.js部署负载均衡方案

## 🧑‍💻 User

nginx与tomcat区别，以及怎么部署写好的nodejs服务器暴露到公网+负载均衡？

---
## 😀 默认助手 (7)

# 一、Nginx 与 Tomcat 的本质区别

核心一句话：**Nginx 是「流量入口 + 静态资源 + 反向代理/负载均衡」的服务器，Tomcat 是「Java Servlet 容器 / 应用服务器」**。二者不是替代关系，生产环境通常 **Nginx 在前，Tomcat 在后**。

| 维度 | Nginx | Tomcat |
|---|---|---|
| 定位 | Web 服务器、反向代理、负载均衡、静态文件服务器 | Java Servlet/JSP 容器、应用服务器 |
| 实现语言/运行时 | C，原生二进制，无 GC | Java，跑在 JVM 上 |
| 并发模型 | 事件驱动 + 异步非阻塞（epoll/kqueue），少量 worker 进程处理海量连接 | 一请求一线程（Tomcat 8.5+ 有 NIO 连接器，但仍受线程池约束，`maxThreads` 默认 200） |
| 单机并发能力 | 轻松上万~十万级长连接，内存占用极小 | 受线程栈 + JVM 堆限制，几千并发就需调优 |
| 启动速度 | 毫秒级 | 秒级到几十秒（Spring Boot 更慢） |
| 动态内容 | **不能执行业务代码**（只能靠 FastCGI 等方式转发给其他进程） | **正是它的本职**：执行 Java Web 应用 |
| 主要能力 | TLS 终止、gzip/br 压缩、静态缓存、限流、灰度、健康检查、负载均衡 | 解析请求、映射 Servlet、管理 Session、连接池、JSP 编译 |
| 典型端口 | 80 / 443 | 8080 / 8009(AJP) |
| 配置风格 | 声明式、极度精简高效 | `server.xml`/`web.xml` 或 Spring Boot 自动配置 |
| 内存占用 | 几十 MB | 几百 MB ~ 数 GB |
| 典型使用 | 所有 Web 架构的最外层入口 | 只服务 Java 后端 |

## 什么时候用哪个

- **纯 Node.js / Python / Go 后端**：只用 Nginx，不需要 Tomcat。
- **Java 后端（Spring Boot）**：Tomcat（或内嵌 Tomcat）+ 前置 Nginx。
- **同机多应用、要 HTTPS、要静态加速、要多实例分流**：Nginx 必上。

<details>
<summary><b>为什么 Java 项目生产环境一定要在前面加 Nginx？（展开）</b></summary>

1. **TLS 终止**：证书、加密握手交给 Nginx，Tomcat 省 CPU，且便于统一管理多域名证书。
2. **静态资源**：JS/CSS/图片由 Nginx 直接返回（sendfile 零拷贝），不占 Tomcat 线程。
3. **慢客户端缓冲**：`proxy_buffering` 让 Tomcat 线程不被慢网络拖住（这是 Tomcat 线程模型的最大痛点）。
4. **统一入口**：一个 80/443 后面挂多个 Tomcat、Node、Python 服务，按 `location`/域名分流。
5. **防护**：限流 `limit_req`、限连接、隐藏版本号、封恶意 UA。
6. **负载均衡与滚动发布**：upstream 摘除节点即可无损重启。

</details>

---

# 二、把 Node.js 服务暴露到公网 + 负载均衡

## 目标架构

```
                    公网用户
                       │  DNS: api.example.com → 你的公网 IP
                       ▼
        ┌──────────────────────────────────┐
        │  云安全组: 只放行 22(限IP) 80 443  │
        └──────────────────────────────────┘
                       ▼
        ┌──────────────────────────────────┐
        │      Nginx :80/:443              │
        │  TLS终止 / gzip / 静态 / 限流      │
        │  upstream 负载均衡 + 健康检查       │
        └───────┬──────────────┬───────────┘
                │              │
        ┌───────▼──────┐ ┌─────▼────────┐
        │ Node 实例 A   │ │ Node 实例 B   │   都只监听 127.0.0.1
        │ 127.0.0.1:3001│ │ 127.0.0.1:3002│   公网无法直连
        └──────────────┘ └──────────────┘
```

**关键原则：Node 进程只绑 `127.0.0.1`，公网只暴露 Nginx 的 80/443。**

> ⚠️ 不要直接把 `node server.js` 监听的 3000 端口通过安全组开放到公网：没有 TLS、没有进程守护、没有限流，几分钟就会被扫描器打满。仅用于临时联调，且用完立刻关。

## 步骤 1：Node 应用做少量改造

```js
// server.js
const express = require('express');
const app = express();

// 1) 端口从环境变量读，方便多实例
const PORT = process.env.PORT || 3000;
// 2) 在 Nginx 后面必须开启，否则 req.ip / req.protocol 拿到的都是反代地址
app.set('trust proxy', 1);

// 3) 健康检查接口，供 Nginx / 云 LB 探活
app.get('/healthz', (req, res) => res.status(200).send('ok'));

// 4) 返回实例信息，用于验证负载均衡是否生效
app.get('/whoami', (req, res) => {
  res.json({ pid: process.pid, host: require('os').hostname(), ip: req.ip });
});

const server = app.listen(PORT, '127.0.0.1', () => {
  console.log(`listening on 127.0.0.1:${PORT}`);
});

// 5) 优雅退出：被摘除时先停止接新连接
process.on('SIGTERM', () => {
  server.close(() => process.exit(0));
  setTimeout(() => process.exit(1), 10000).unref();
});
```

## 步骤 2：进程守护 + 多实例

负载均衡的前提是**有多个进程/多台机器**。选一种：

**方案 A：PM2 cluster（单机多核，最省事）**

```js
// ecosystem.config.js
module.exports = {
  apps: [{
    name: 'api',
    script: './server.js',
    instances: 'max',           // 用满 CPU 核数
    exec_mode: 'cluster',       // 多进程共享同一端口(3000)
    max_memory_restart: '600M',
    out_file: '/var/log/nodeapp/out.log',
    error_file: '/var/log/nodeapp/err.log',
    env: { NODE_ENV: 'production', PORT: 3000 }
  }]
};
```

```bash
pm2 start ecosystem.config.js
pm2 save && pm2 startup   # 开机自启
pm2 logs api              # 看日志
```

**方案 B：systemd + 多端口多实例（配合 Nginx 做真正的 upstream 负载均衡）**

```ini
# /etc/systemd/system/nodeapp@.service   模板服务，@ 后面的数字即端口
[Unit]
Description=Node App on port %i
After=network.target

[Service]
Type=simple
User=nodeapp
WorkingDirectory=/opt/app
Environment=NODE_ENV=production
Environment=PORT=%i
ExecStart=/usr/bin/node /opt/app/server.js
Restart=always
RestartSec=3
LimitNOFILE=65535
StandardOutput=append:/var/log/nodeapp/%i.log
StandardError=append:/var/log/nodeapp/%i.err.log

[Install]
WantedBy=multi-user.target
```

```bash
for p in 3001 3002 3003; do sudo systemctl enable --now nodeapp@$p; done
```

<details>
<summary><b>cluster 模式 vs 多端口多实例，该选哪个？</b></summary>

| | PM2 cluster | 多端口 + upstream |
|---|---|---|
| 本地多核利用 | 原生支持，单端口 | 每实例一端口 |
| Nginx 是否能真正分发 | ❌ 请求全打到 3000，由 Node 内核 `SO_REUSEPORT` 分发 | ✅ upstream 逐个轮询，可加权、可摘节点 |
| 滚动发布/摘流 | 较粗（reload 整体） | 细粒度，改 upstream 后 `nginx -s reload` |
| 跨机器扩展 | 需再多层 | 天然，upstream 写各机器 IP 即可 |
| 适用 | 单机小项目 | 生产推荐（尤其配合容器/多机） |

> 结论：**单机先跑起来用 cluster；要真正的负载均衡与灰度，用多实例 + upstream（多机同理，只是把 `127.0.0.1` 换成内网 IP）。**

</details>

## 步骤 3：Nginx 反向代理 + 负载均衡

先装 Nginx：

```bash
# Debian/Ubuntu
sudo apt update && sudo apt install -y nginx
# CentOS/Rocky
sudo dnf install -y nginx && sudo systemctl enable --now nginx
```

`/etc/nginx/nginx.conf` 的 `http {}` 块内加入（WebSocket 升级必需）：

```nginx
map $http_upgrade $connection_upgrade {
    default upgrade;
    ''      close;
}

# 限流：每 IP 每秒 20 个请求，突发 40
limit_req_zone $binary_remote_addr zone=apilimit:10m rate=20r/s;
```

`/etc/nginx/conf.d/api.conf`：

```nginx
upstream node_backend {
    least_conn;                              # 最少连接，适合长短混合请求；纯轮询则删掉这行

    server 127.0.0.1:3001 max_fails=3 fail_timeout=10s;
    server 127.0.0.1:3002 max_fails=3 fail_timeout=10s;
    server 127.0.0.1:3003 max_fails=3 fail_timeout=10s backup;   # 备用节点（可删）

    keepalive 64;                            # 与后端保持长连接，减少握手开销
}

# HTTP → HTTPS
server {
    listen 80;
    server_name api.example.com;

    # 证书申请时的 ACME 校验放行
    location /.well-known/acme-challenge/ { root /var/www/certbot; }

    location / { return 301 https://$host$request_uri; }
}

server {
    listen 443 ssl;
    http2 on;                                # nginx ≥1.25.1 写法；旧版用 listen 443 ssl http2;
    server_name api.example.com;

    ssl_certificate     /etc/letsencrypt/live/api.example.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/api.example.com/privkey.pem;
    ssl_protocols       TLSv1.2 TLSv1.3;
    ssl_session_cache   shared:SSL:10m;

    server_tokens off;                       # 隐藏版本号
    client_max_body_size 20m;                # 上传大小，默认 1m 容易 413

    access_log /var/log/nginx/api.access.log;
    error_log  /var/log/nginx/api.error.log warn;

    # 静态资源由 Nginx 直出，不经过 Node
    location /static/ {
        alias /var/www/app/public/;
        expires 30d;
        add_header Cache-Control "public, immutable";
        access_log off;
    }

    location / {
        limit_req zone=apilimit burst=40 nodelay;

        proxy_pass http://node_backend;
        proxy_http_version 1.1;              # keepalive 与 WebSocket 都需要

        # WebSocket / SSE 升级
        proxy_set_header Upgrade    $http_upgrade;
        proxy_set_header Connection $connection_upgrade;

        # 真实客户端信息，Node 端配合 trust proxy 使用
        proxy_set_header Host              $host;
        proxy_set_header X-Real-IP         $remote_addr;
        proxy_set_header X-Forwarded-For   $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_set_header X-Forwarded-Host  $host;

        # 超时
        proxy_connect_timeout 5s;
        proxy_send_timeout    60s;
        proxy_read_timeout    60s;           # SSE/长轮询要调大，如 3600s

        # 缓冲，避免慢客户端占用 Node
        proxy_buffering on;
        proxy_buffers 16 16k;
        proxy_busy_buffers_size 32k;

        # 不让后端返回的重定向泄露内网地址
        proxy_redirect off;
    }
}
```

校验并生效：

```bash
sudo nginx -t && sudo systemctl reload nginx
```

## 步骤 4：HTTPS 证书（Let's Encrypt 免费）

```bash
sudo apt install -y certbot python3-certbot-nginx
sudo certbot --nginx -d api.example.com --redirect
# 自动续期测试
sudo certbot renew --dry-run
```

## 步骤 5：域名与网络放行

1. DNS 添加 `A` 记录：`api.example.com → 服务器公网 IP`。
2. 云厂商**安全组/防火墙**只放行 `80`、`443`，`22` 限制来源 IP；**不要放行 3001~3003**。
3. 服务器本机防火墙：

```bash
sudo ufw allow 80,443/tcp && sudo ufw allow from <你的IP> to any port 22 && sudo ufw enable
# 或 firewalld
sudo firewall-cmd --permanent --add-service={http,https} && sudo firewall-cmd --reload
```

4. 若系统开了 SELinux（CentOS/Rocky），反代到本地端口需放行，否则 502：
```bash
sudo setsebool -P httpd_can_network_connect 1
```

## 步骤 6：验证负载均衡真的生效

```bash
# 健康检查
curl -i https://api.example.com/healthz

# 连续请求，观察 pid 是否轮转
for i in $(seq 1 9); do curl -s https://api.example.com/whoami; echo; done

# 验证真实 IP 透传
curl -s https://api.example.com/whoami
```

如果 `/whoami` 一直返回同一个 pid，说明 Nginx 没在做分发（多半是只起了 1 个实例，或者用了 PM2 cluster 单端口模式）。

## 负载均衡策略怎么选

| 策略 | 配置 | 适用 |
|---|---|---|
| 轮询（默认） | 直接列 `server` | 实例同质、请求耗时均匀 |
| 加权轮询 | `server 10.0.0.2:3000 weight=3;` | 机器配置不同（新老机器混部） |
| 最少连接 | `least_conn;` | 请求耗时长且差异大（推荐 Node 场景） |
| IP 哈希 | `ip_hash;` | 必须会话粘滞且无法共享 Session |
| 一致性哈希 | `hash $request_uri consistent;` | 提高缓存命中率（需第三方模块/商业版） |

**健康检查**：开源版 Nginx 靠被动检查（`max_fails` + `fail_timeout`），失败 3 次后 10 秒内不再分配；10 秒后自动重试。要主动探测请用云负载均衡或 Nginx Plus / OpenResty。

<details>
<summary><b>多机部署（水平扩容）怎么改？</b></summary>

```nginx
upstream node_backend {
    least_conn;
    server 10.0.1.11:3000 max_fails=3 fail_timeout=10s;
    server 10.0.1.12:3000 max_fails=3 fail_timeout=10s;
    server 10.0.1.13:3000 max_fails=3 fail_timeout=10s;
    keepalive 128;
}
```

要点：
- 用**内网 IP / 私有网络**，不要走公网（省流量费、低延迟、安全）。
- Nginx 所在机器本身也是单点 → 上层再挂云 SLB/CLB，或 DNS 轮询 + Keepalived VIP 双机。
- 各机器时钟用 NTP 同步，日志集中收集（Loki/ELK）。

</details>

<details>
<summary><b>Docker Compose 一键版（含 Nginx）</b></summary>

```yaml
# docker-compose.yml
services:
  api:
    build: .
    environment:
      - NODE_ENV=production
      - PORT=3000
    expose: ["3000"]            # 只对内部网络，不映射到宿主机
    restart: unless-stopped
    deploy:
      replicas: 3               # 3 个实例；Nginx 的 upstream 用服务名解析

  nginx:
    image: nginx:1.27-alpine
    ports: ["80:80", "443:443"]
    volumes:
      - ./nginx/conf.d:/etc/nginx/conf.d:ro
      - ./nginx/certs:/etc/nginx/certs:ro
      - ./public:/var/www/app/public:ro
    depends_on: [api]
    restart: unless-stopped
```

Docker 内置 DNS 会返回多个容器 IP，但 Nginx 开源版**启动时只解析一次**，容器扩缩容后会失效。稳妥做法：
- upstream 用 `server api:3000 resolve;`（需 Nginx ≥1.27.3 + `zone`，或使用 OpenResty / 第三方 resolve 模块），或
- 每个实例固定端口 + `network_mode: host`，或
- 直接用 Docker Swarm/K8s 的负载均衡，Nginx 只做入口。

最省心还是：**容器各自映射 3001/3002/3003，Nginx upstream 写死这几个端口。**

</details>

<details>
<summary><b>上线后必查的 7 个问题与排查表</b></summary>

| 现象 | 常见原因 | 处理 |
|---|---|---|
| **502 Bad Gateway** | Node 没起 / 端口不对 / 绑在 `127.0.0.1` 但 Nginx 在另一台机器 / SELinux 拦截 | `ss -lntp \| grep 300`、`journalctl -u nodeapp@3001`、`setsebool httpd_can_network_connect 1` |
| **504 Gateway Timeout** | 接口耗时 > `proxy_read_timeout`（默认 60s） | 调大超时或改异步任务；排查 Node 阻塞（CPU 密集、同步 IO） |
| **413 Request Entity Too Large** | `client_max_body_size` 默认 1m | 调大，或在 Node 层用流式上传 |
| **WebSocket 握手失败** | 缺 `Upgrade`/`Connection` 头，或用了 `proxy_http_version 1.0` | 按上面配置补 `map` 与 `proxy_http_version 1.1` |
| **拿到的客户端 IP 全是 127.0.0.1** | 未透传 `X-Forwarded-For` / 未设 `trust proxy` | 两边都配 |
| **SSE 流式响应被攒着不发** | `proxy_buffering on` | 该 location 设 `proxy_buffering off;` |
| **HTTPS 跳转后变成了 http** | 应用生成了绝对 URL 且不知道是 https | `X-Forwarded-Proto` + 框架的 proxy 信任配置 |

</details>

<details>
<summary><b>安全与容量补充</b></summary>

**安全加固**
- Node 用非 root 用户运行（`User=nodeapp`），并 `LimitNOFILE=65535`。
- Nginx 加 `limit_conn_zone` 限制单 IP 并发连接。
- 关闭不必要的 HTTP 方法、过滤异常 UA；`add_header X-Content-Type-Options nosniff`、`X-Frame-Options DENY`。
- 后端不返回堆栈信息，`NODE_ENV=production`。
- 定期 `npm audit`，镜像固定版本、非 `latest`。

**容量估算（Little 定律）**

并发连接数 $L$、请求到达率 $\lambda$（req/s）、平均响应时间 $W$（秒）满足：

$$L = \lambda \times W$$

例如 $\lambda = 2000\ \text{req/s}$、$W = 0.05\ \text{s}$，则平均并发 $L = 100$。再按每实例安全承载量（Node 单进程建议 < 300 并发常驻）反推实例数 $N = \lceil L / 300 \rceil$，并留 50% 余量做峰值与故障冗余。

**有状态问题**：Node 多实例后 `session`/内存缓存会串。优先用 **JWT 无状态**；必须会话时用 **Redis 共享 Session**，或 Nginx `ip_hash`（仅临时方案，扩缩容会打乱）。

</details>

---

## 最小可用清单（照着做就能上公网）

1. Node 改读 `PORT`、加 `/healthz`、开 `trust proxy`。
2. 起 3 个实例：`systemctl enable --now nodeapp@{3001,3002,3003}`（或 `pm2 start -i max`）。
3. Nginx `upstream` 指向上面的端口 + `proxy_pass` + WebSocket 头。
4. `certbot --nginx` 拿证书，80 自动跳 443。
5. 安全组只开 80/443/22，Node 端口一律不对外。
6. `for i in $(seq 9); do curl -s https://域名/whoami; done` 看 pid 是否轮转 → 负载均衡生效。
7. 配好日志与 `pm2 logs` / `journalctl`，出问题先看 Nginx error_log 还是 Node 日志。
