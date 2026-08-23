# PVE多拨公网给VM

### 🧑‍💻 User

有公网你可以pppoe多播然后去分给vm，这在pve上到底要怎么实现？

---

### 🤖 Assistant

在 PVE 上实现 **PPPoE 多拨 → 公网 IP 分配给虚拟机**，核心思路是让宿主机同时拨多个 PPPoE 会话，为每次会话获取一个独立的公网 IP，再通过 NAT 或路由技术把这些公网 IP “送给”对应的虚拟机使用。

根据实际操作和稳定性，推荐两种主流方案：

- **方案一：VM 自行拨号**（PVE 做透明桥，每个 VM 自己跑 PPPoE 客户端）
  最简单，没有端口转发困扰，但需要虚拟机系统支持 PPPoE，部分云镜像不易配置。

- **方案二：宿主机集中多拨 + 1:1 NAT 映射给 VM**
  灵活、可控，虚拟机只需普通内网 IP，外部通过映射和公网 IP 通信，是生产环境常用做法。

下面重点介绍 **方案二** 的完整实现细节（兼容 Debian/PVE 7.x/8.x），同时会给出方案一的简洁说明。

---

## 方案二：PVE 宿主多拨 + 静态 1:1 NAT 映射

### 整体拓扑
```
Internet ── 光猫桥接 ── 物理网口 eth0 (PVE)
                 │
          ┌──────┴──────┐
          │ macvlan0    │ macvlan1    ...  (不同 MAC 的虚拟接口)
          │ ppp0 (公网IP1)  │ ppp1 (公网IP2)
          └──────┬──────┘
                 │ 宿主机内核 (路由 + NAT)
          ┌──────┴──────┐
          │ vmbr0 (内网网桥: 10.0.0.0/24)
          ├─────────────┤
          │ VM-1 10.0.0.101   │ VM-2 10.0.0.102
          └───────────────────┘
```
每个 VM 的内网 IP 通过 `iptables NETMAP` 或 `SNAT/DNAT` 与一个 ppp 接口的**完整公网 IP**进行双向转换，相当于 VM 独享一个公网地址。

---

### 第一步：准备物理接口与 macvlan

确认你用来拨号的物理网口（假设为 `eth0`）没有绑定到任何 vmbr。若日需保留原有管理网桥，可用另一个独立网口。

1. **创建多个 macvlan 子接口**（每个接口用不同 MAC，避免运营商限制单 MAC 会话数）：

```bash
# 创建 4 个 macvlan 接口，假设你需要 4 拨
for i in 0 1 2 3; do
    ip link add link eth0 macvlan$i type macvlan mode bridge
done
```

2. **为每个子接口设置随机的或手动指定的 MAC 地址**（防止冲突，建议手动指定）：

```bash
ip link set macvlan0 address 02:01:02:03:04:00 up
ip link set macvlan1 address 02:01:02:03:04:01 up
ip link set macvlan2 address 02:01:02:03:04:02 up
ip link set macvlan3 address 02:01:02:03:04:03 up
```

> 注意：`eth0` 本身不要配置 IP，保持 UP 状态即可。

**持久化**：将上述命令写入 `/etc/network/interfaces` 或 PVE 的 `interfaces` 文件中，例如：

```
auto macvlan0
iface macvlan0 inet manual
	pre-up ip link add link eth0 macvlan0 type macvlan mode bridge
	pre-up ip link set macvlan0 address 02:01:02:03:04:00
	up ip link set macvlan0 up
	post-down ip link delete macvlan0
# 其余类似
```

---

### 第二步：安装 PPPoE 客户端并配置多拨

```bash
apt update
apt install ppp pppoe
```

不使用 `pppoeconf` 自动向导，直接手动创建配置文件。每个拨号会话对应一个 peer 文件和一次 `pppd` 调用。

1. **创建认证信息文件** `/etc/ppp/pap-secrets` （或 `chap-secrets`）：

```
# 客户端    服务器    密码             IP地址
"your_pppoe_user" * "your_pppoe_pass" *
```

2. **为每个会话创建 peer 文件**，例如 `/etc/ppp/peers/pppoe0`：

```
# /etc/ppp/peers/pppoe0
plugin rp-pppoe.so
# 绑定到对应的 macvlan 接口
nic-macvlan0
user "your_pppoe_user"
noipdefault
defaultroute
replacedefaultroute
hide-password
noauth
persist
maxfail 0
# 使用独立单元号 0，产生的 ppp 接口为 ppp0
unit 0
# 下面两行可以避免把默认路由覆盖到 ppp0
nodefaultroute
# 不修改 DNS (可以用本地 DNS)
usepeerdns
```

对于第二个连接 `pppoe1`，复制并修改：

```
nic-macvlan1
unit 1
nodefaultroute
...
```

同理，创建 `pppoe2`, `pppoe3`。

> **关键参数**：
> - `nic-<interface>` 指定用于拨号的网络接口。
> - `unit N` 指定 ppp 接口编号，如 ppp0、ppp1。
> - `nodefaultroute` 防止多个 ppp 同时添加默认路由导致混乱。
> - `persist maxfail 0` 保持连接不断重连。

3. **启动拨号**（可以用 `pon` 或 `pppd call`）：

```bash
pon pppoe0
pon pppoe1
# 或
pppd call pppoe0
```

查看状态：`ip addr show ppp0`，`pppstats`。

---

### 第三步：配置内网桥 vmbr0 和虚拟机

确保 PVE 有内网网桥 `vmbr0`（假如 IP 段为 `10.0.0.0/24`，宿主机 IP `10.0.0.1`）。所有需要公网映射的 VM 网卡都桥接到 `vmbr0`，并配置静态内网 IP，例如：

- VM1：`10.0.0.101/24`，网关 `10.0.0.1`
- VM2：`10.0.0.102/24`

---

### 第四步：配置 1:1 NAT（NETMAP）绑定公网 IP 与 VM 内网 IP

假设分配规则：
- ppp0 的公网 IP → VM1 (10.0.0.101)
- ppp1 的公网 IP → VM2 (10.0.0.102)

因为 PPPoE 拨号后 IP 是动态获得的，需要脚本动态获取 IP 并更新 iptables 规则。推荐使用 `ip-up` 脚本，当 ppp 连接建立后自动执行。

**创建 `/etc/ppp/ip-up.d/99-netmap-vm.sh`**：

```bash
#!/bin/bash
# 此脚本由 pppd 在连接建立后调用，参数：接口名 设备 速度 本地IP 远程IP 网络掩码
PPP_IFACE=$1
LOCAL_IP=$4
REMOTE_IP=$5

VM_IP=""   # 根据 ppp 接口决定

case "$PPP_IFACE" in
    ppp0) VM_IP="10.0.0.101" ;;
    ppp1) VM_IP="10.0.0.102" ;;
    ppp2) VM_IP="10.0.0.103" ;;
    ppp3) VM_IP="10.0.0.104" ;;
esac

if [ -n "$VM_IP" ]; then
    # 清除本接口旧的 NAT 规则（如果有）
    iptables -t nat -D PREROUTING -i $PPP_IFACE -j NETMAP --to $VM_IP 2>/dev/null
    iptables -t nat -D POSTROUTING -s $VM_IP -j NETMAP --to $LOCAL_IP 2>/dev/null

    # 添加双向 NETMAP
    iptables -t nat -A PREROUTING -i $PPP_IFACE -j NETMAP --to $VM_IP
    iptables -t nat -A POSTROUTING -s $VM_IP -j NETMAP --to $LOCAL_IP

    # 同时放行转发
    iptables -A FORWARD -i $PPP_IFACE -d $VM_IP -j ACCEPT
    iptables -A FORWARD -s $VM_IP -o $PPP_IFACE -j ACCEPT
fi
```

赋予执行权限：`chmod +x /etc/ppp/ip-up.d/99-netmap-vm.sh`

同样，在 `/etc/ppp/ip-down.d/99-netmap-vm.sh` 中编写清理规则的动作（结构与上类似，只是删除规则）。

启用 IP 转发：`sysctl -w net.ipv4.ip_forward=1`，并写入 `/etc/sysctl.conf`。

**原理说明**：
`NETMAP` 是一种一对一的 NAT，它把进入 `ppp0` 的报文目的 IP（即公网 IP）直接替换为 `10.0.0.101`，同时将来自 `10.0.0.101` 的报文源 IP 替换为 `ppp0` 的公网 IP。VM 完全无感知，外面就认为这个公网 IP 就是 VM 的真实地址，包括所有端口、协议都直接映射。

> 如果你的运营商只给 `/32` 地址并且网关指向对端，这种 NAT 模式完全透明，除了某些对源 IP 有严格检查的 UDP 服务需要额外处理，绝大部分应用都正常工作。

---

### 第五步：日常维护与断线重拨

- **自动重拨**：peer 文件中的 `persist` 已经保证断线自动重连。重连后 `ip-up` 脚本会触发，更新 iptables 规则。
- **多拨启动**：建议使用 systemd 服务保证在宿主机启动时一并启动所有 pppoe 连接。例如创建 `/etc/systemd/system/pppoe-multi@.service`：

```
[Unit]
Description=PPPoE connection for %i
After=network.target

[Service]
Type=simple
ExecStart=/usr/sbin/pppd call %i
Restart=always

[Install]
WantedBy=multi-user.target
```

启动：`systemctl enable --now pppoe-multi@pppoe0 pppoe-multi@pppoe1 ...`

- **检查状态**：`ip addr show ppp0`、`iptables -t nat -L -n -v`。

---

## 方案一：VM 自行拨号（透明桥接）

如果希望虚拟机直接获得公网 IP，无需在宿主机做 NAT，操作更简单但要求 VM 能运行 PPPoE 客户端。

1. 在 PVE 上为每个需要拨号的 VM 创建一个独立的 `Linux Bridge`（例如 `vmbr1`、`vmbr2`），并把**物理拨号网口**（例如 eth0.10 或其他）加入到该 bridge 中（或者使用 openvswitch 进行端口隔离）。

2. 物理网口上可能需要打上运营商 VLAN 或直接使用无 VLAN 的桥接，确保广播包能够到达 VM 的网卡。

3. VM 的网桥端口设置为 `virtio` / `e1000`，然后在 VM 内部按照常规方式（`pppoeconf`、`pppd`）拨号，即可获取公网 IP。

缺点：一个物理网口同时分给多个 VM 拨号时，需要处理二层广播域冲突和 MAC 限制，很难在同一物理端口上让多个 VM 同时拨号且获得不同 IP（需要 macvlan 桥或 SR-IOV）。因此若要同时多 VM 独立拨号，仍需配合 PVE 宿主机的 macvlan 或使用支持 SR-IOV 的网卡将 VF 直通给 VM，后者较为复杂。

---

## 常见问题排查

- **拨号失败**：检查 macvlan 接口是否 UP，是否与 eth0 在同一 VLAN。部分运营商需要 VLAN 标签，可创建带 VLAN 的子接口，再建 macvlan。
- **只有第一个拨号成功**：修改 MAC 地址，某些局端严格限制相同 MAC 同时在线。
- **NAT 后端口不通**：确认 `ip_forward=1`，iptables FORWARD 链策略允许，且在 `/etc/ppp/ip-up` 中正确添加了 NETMAP。
- **公网 IP 到达 VM 后回程路由不对**：因为 VM 的默认网关设为宿主机内网 IP（10.0.0.1），需要宿主机启用 **proxy_arp** 或将出向路由正确 SNAT。`NETMAP` 已经处理了源地址转换，所以基本无需额外路由。

按照上面的方案二，你可以在 PVE 上稳定实现多拨公网 IP 共享给多台虚拟机，每个虚拟机就像独占一个公网宽带出口一样。