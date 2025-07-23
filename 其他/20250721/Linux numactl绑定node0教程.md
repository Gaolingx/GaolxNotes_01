# Linux numactl绑定node0教程

### 🧑‍💻 User

linux如何用numactl绑定node0？

---

### 🤖 Assistant

在Linux系统中，使用`numactl`工具将程序绑定到特定的NUMA节点（如node0）可以通过以下步骤实现：

### 1. **基本命令**
```bash
numactl --cpunodebind=0 --membind=0 <你的程序> [程序参数]
```
- `--cpunodebind=0`：将进程的CPU绑定到node0的物理核心。
- `--membind=0`：强制内存分配只使用node0的内存。

### 2. **简化写法**
```bash
numactl -N 0 -m 0 <你的程序> [程序参数]
```
- `-N 0`：等同于`--cpunodebind=0`
- `-m 0`：等同于`--membind=0`

### 3. **仅绑定CPU节点（自动使用本地内存）**
```bash
numactl --cpunodebind=0 --localalloc <你的程序> [程序参数]
```
- `--localalloc`：在运行进程的NUMA节点上分配内存（与`--cpunodebind`配合时，内存也会在node0分配）。

---

### **实际示例**
假设要运行程序`./myapp`并绑定到node0：
```bash
numactl -N 0 -m 0 ./myapp arg1 arg2
```

---

### **验证绑定结果**
1. **查看进程的NUMA状态**：
   ```bash
   numastat -p <PID>
   ```
   输出中`N0`列的百分比应为100%。

2. **使用`taskset`检查CPU亲和性**：
   ```bash
   taskset -cp <PID>
   ```
   显示的CPU核心列表应属于node0（通过`numactl --hardware`可查看node0包含哪些核心）。

---

### **注意事项**
- **节点有效性**：先用`numactl --hardware`确认node0存在。
- **内存不足**：如果node0内存不足，使用`--membind`可能导致内存分配失败，此时可移除`-m 0`改用`--localalloc`。
- **兼容性**：部分程序可能内置NUMA策略（如MySQL），需在程序配置中调整。

### **查看NUMA拓扑**
```bash
numactl --hardware
```
输出示例：
```
available: 2 nodes (0-1)
node 0 cpus: 0 1 2 3
node 0 size: 16384 MB
node 1 cpus: 4 5 6 7
node 1 size: 16384 MB
```
这里`node 0`包含CPU核心0-3，内存16GB。

通过以上命令，即可将程序严格绑定到NUMA node0的CPU和内存资源上。