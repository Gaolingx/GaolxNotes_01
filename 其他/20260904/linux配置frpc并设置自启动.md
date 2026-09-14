# linux配置frpc并设置自启动

### 🧑‍💻 User

linux如何配置frpc并设置自启动？

---

### 🤖 Assistant

# Linux 配置 frpc 并设置自启动

frp（Fast Reverse Proxy）是一款流行的内网穿透工具，`frpc` 是它的客户端，运行在内网机器上。下面从安装、配置到自启动一步步说明。

---

## 一、下载并安装 frp

### 1. 下载对应架构的包

到 [frp GitHub Releases](https://github.com/fatedier/frp/releases) 下载与系统架构匹配的版本（`amd64` / `arm64` 等）：

```bash
# 以 amd64 为例，版本号请换成最新的
cd /usr/local/src
wget https://github.com/fatedier/frp/releases/download/v0.61.0/frp_0.61.0_linux_amd64.tar.gz
tar -zxvf frp_0.61.0_linux_amd64.tar.gz
```

### 2. 移动到规范目录

```bash
sudo mkdir -p /usr/local/frp
sudo cp frp_0.61.0_linux_amd64/frpc /usr/local/frp/
sudo cp frp_0.61.0_linux_amd64/frpc.toml /usr/local/frp/
```

> **版本提示**：frp 从 `v0.52.0` 起把配置文件从 INI 格式（`frpc.ini`）改成了 TOML 格式（`frpc.toml`）。下面以新版 TOML 为准，旧版只需把后缀和字段名稍作调整。

---

## 二、配置 frpc.toml

编辑配置文件：

```bash
sudo vim /usr/local/frp/frpc.toml
```

### 1. 基础示例（TCP 穿透 SSH / 内网服务）

```toml
# 服务器连接配置
serverAddr = "your-server-ip.com"
serverPort = 7000

# 认证令牌（需与 frps 端 token 一致）
auth.method = "token"
auth.token = "your-token-here"

# 可选：日志
log.to = "/var/log/frpc.log"
log.level = "info"
log.maxDays = 3

# 代理配置
[[proxies]]
name = "ssh"
type = "tcp"
localIP = "127.0.0.1"
localPort = 22
remotePort = 6000
```

### 2. 常用代理类型示例

```toml
# HTTP/HTTPS 内网穿透（通过自定义域名访问）
[[proxies]]
name = "web"
type = "http"
localIP = "127.0.0.1"
localPort = 8080
customDomains = ["www.example.com"]

# HTTPS 穿透
[[proxies]]
name = "web-https"
type = "https"
localIP = "127.0.0.1"
localPort = 8443
customDomains = ["www.example.com"]

# UDP 穿透（如 DNS、游戏）
[[proxies]]
name = "dns"
type = "udp"
localIP = "8.8.8.8"
localPort = 53
remotePort = 6001
```

### 3. 多客户端配置说明

每台内网机器可配置一个唯一的 `user` 标识（新版可选）：

```toml
user = "home-pc"
serverAddr = "your-server-ip.com"
serverPort = 7000
```

---

## 三、手动运行测试

先手动运行确认配置无误：

```bash
# 前台运行，直接看日志
/usr/local/frp/frpc -c /usr/local/frp/frpc.toml

# 后台运行（测试用）
/usr/local/frp/frpc -c /usr/local/frp/frpc.toml &
```

能正常连接后，`Ctrl+C` 停掉，接下来配置自启动。

---

## 四、使用 systemd 设置自启动（推荐）

### 1. 创建服务文件

```bash
sudo vim /etc/systemd/system/frpc.service
```

写入以下内容：

```ini
[Unit]
Description=frp Client (frpc)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
# 以 root 运行；如需低权限可改为专用用户
User=root
Restart=on-failure
RestartSec=5s
# 如需后台守护模式可加 -n 参数；新版默认前台运行即可
ExecStart=/usr/local/frp/frpc -c /usr/local/frp/frpc.toml
ExecStop=/bin/kill -TERM $MAINPID

[Install]
WantedBy=multi-user.target
```

### 2. 重载并启动服务

```bash
# 重新加载 systemd 配置
sudo systemctl daemon-reload

# 设置开机自启动
sudo systemctl enable frpc

# 立即启动
sudo systemctl start frpc

# 查看运行状态
sudo systemctl status frpc
```

### 3. 常用管理命令

```bash
sudo systemctl stop frpc      # 停止
sudo systemctl restart frpc   # 重启
sudo systemctl disable frpc   # 取消开机自启
journalctl -u frpc -f         # 实时查看日志
```

---

## 五、其他自启动方式（可选）

### 1. cron 方式（最简单）

```bash
crontab -e
# 添加下面一行，开机 1 分钟后启动
@reboot sleep 60 && /usr/local/frp/frpc -c /usr/local/frp/frpc.toml > /dev/null 2>&1 &
```

### 2. init.d 脚本（老式 SysVinit 系统）

```bash
sudo vim /etc/init.d/frpc
```

```bash
#!/bin/bash
### BEGIN INIT INFO
# Provides:          frpc
# Required-Start:    $network $remote_fs
# Default-Start:     2 3 4 5
# Default-Stop:      0 1 6
### END INIT INFO

case "$1" in
  start)
    /usr/local/frp/frpc -c /usr/local/frp/frpc.toml &
    ;;
  stop)
    pkill -f "frpc -c /usr/local/frp/frpc.toml"
    ;;
  restart)
    $0 stop
    sleep 1
    $0 start
    ;;
  *)
    echo "Usage: $0 {start|stop|restart}"
    exit 1
    ;;
esac
exit 0
```

```bash
sudo chmod +x /etc/init.d/frpc
sudo update-rc.d frpc defaults   # Debian/Ubuntu
# 或 sudo chkconfig frpc on       # CentOS/RHEL
```

---

## 六、排查问题

| 现象 | 排查方法 |
|------|---------|
| 连接不上服务器 | 检查 `serverAddr`/`serverPort` 是否正确，服务器防火墙是否放行端口 |
| 认证失败 | 确认 `auth.token` 与 `frps` 端完全一致 |
| 端口占用 | 查看 `remotePort` 是否被占用：`ss -lntp` |
| 域名不通 | 确认域名已解析到服务器 IP，且 `frps` 已配置 `vhostHTTPPort` |
| 日志查看 | `journalctl -u frpc -f` 或 `tail -f /var/log/frpc.log` |

---

**总结**：核心流程就是「下载 → 配置 `frpc.toml` → 手测 → systemd 服务自启动」。systemd 是目前 Linux 发行版最通用、最推荐的方式，支持自动重启、日志管理和开机自启。

如果你能告诉我具体的穿透场景（SSH、网站、数据库等）和 Linux 发行版，我可以给出更针对性的配置示例。