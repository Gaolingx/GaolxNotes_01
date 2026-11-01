# 硬球型结构抽象  自适应粗粒化  稀薄气体计算  NS方程极限

## 🧑‍💻 User

设计一个类似硬球模型一样的结构，并定义结构的操作，最后一样可以获得ns方程。

---
## 😀 默认助手 (7)

可以。关键不是“球”，而是硬球模型背后的结构：**局部二体碰撞 + 守恒律 + 可逆性/细致平衡 + 各向同性 + 尺度分离**。把这五条抽象出来，就得到一个“硬球型结构”；硬球只是其中一种碰撞核。只要结构满足这些条件，用同一套 Chapman–Enskog 机器作二阶截断，就能得到 Navier–Stokes 方程。

下面给出一个完整设计。

---

## 1. 抽象结构

定义结构
\[
\mathfrak S=(\Gamma,\mathsf{Str},\mathsf{Col},\mathsf{Obs},\mathsf{Avg},\mathsf{Lim}).
\]

### 1.1 状态空间

连续版本：
\[
\Gamma_N=\left\{(x_i,v_i,m_i)_{i=1}^N:\ x_i\in\Omega\subset\mathbb R^3,\ v_i\in\mathbb R^3,\ |x_i-x_j|\ge 2a\right\}.
\]

离散版本：
\[
\Gamma_L=\left\{n:\mathcal L\to\{0,1\}^{\mathcal V}\right\},
\]
其中 \(\mathcal L\) 是格子，\(\mathcal V=\{c_1,\dots,c_q\}\) 是离散速度集。

### 1.2 操作

| 操作 | 记号 | 定义 | 作用 |
|---|---|---|---|
| 自由飞行 | \(S_\tau\) | \(x_i\mapsto x_i+\tau v_i\) | 平流 |
| 碰撞 | \(C\) | \((v_i,v_j)\mapsto(v_i',v_j')\) | 局部重分配速度 |
| 步进 | \(\Phi_\tau=C\circ S_\tau\) | 先流后碰 | 一步演化 |
| 观测 | \(O_\phi\) | \(\langle\phi\rangle\) | 提取宏观场 |
| 平均 | \(A_\epsilon\) | 空间/系综平均 | 粗粒化 |
| 极限 | \(\mathcal L_\epsilon\) | \(\epsilon\to0\) | 得到 PDE |

碰撞关系定义为
\[
\mathcal R=\left\{(v,w;v',w'):\ v'+w'=v+w,\quad |v'|^2+|w'|^2=|v|^2+|w|^2\right\}.
\]
硬球是其中一个特例。等质量硬球碰撞为
\[
v'=v-\big[(v-w)\cdot n\big]n,\qquad
w'=w+\big[(v-w)\cdot n\big]n,
\]
其中
\[
n=\frac{x_i-x_j}{|x_i-x_j|},\qquad (v-w)\cdot n<0.
\]
不等质量时把系数换成 \(\frac{2m_j}{m_i+m_j}\) 和 \(\frac{2m_i}{m_i+m_j}\) 即可。

### 1.3 结构律

一个“硬球型结构”必须满足：

1. **局部性**：碰撞只依赖有限邻域内的粒子。
2. **守恒性**：
\[
\sum_i m_i,\qquad \sum_i m_iv_i,\qquad \sum_i \frac12m_i|v_i|^2
\]
在碰撞下不变。
3. **微观可逆性**：
\[
R\circ\Phi_t\circ R=\Phi_{-t},\qquad R(x,v)=(x,-v).
\]
4. **H 定理**：
\[
H(f)=\int f\log f\,dv,\qquad \frac{dH}{dt}\le0.
\]
5. **各向同性**：碰撞算子与空间旋转交换。
6. **尺度分离**：存在小 Knudsen 数
\[
Kn=\frac{\lambda}{L}\ll1,
\]
其中 \(\lambda\) 是平均自由程。

---

## 2. 具体实现

### 2.1 连续硬球：事件驱动

```text
step(state):
    t_c = 所有粒子对的最早碰撞时间
    把所有粒子推进到 t_c
    解决所有同时发生的碰撞
    重复
```

这是最贴近物理的版本，适合作为“基准模型”。

### 2.2 格子硬球：可计算版本

取格子 \(\mathcal L=h\mathbb Z^d\)，速度集 \(\mathcal V=\{c_1,\dots,c_q\}\)，占据数
\[
n_i(x,t)\in\{0,1\}.
\]

操作定义为：

**Streaming：**
\[
n_i^*(x+c_i\Delta t,t+\Delta t)=n_i(x,t).
\]

**Collision：** 在每个格点施加局部双射
\[
C_x:\{0,1\}^q\to\{0,1\}^q,
\]
要求
\[
\sum_i C_x(n)_i=\sum_i n_i,\qquad
\sum_i C_x(n)_i c_i=\sum_i n_i c_i,\qquad
\sum_i C_x(n)_i |c_i|^2=\sum_i n_i |c_i|^2.
\]

**平均场：**
\[
f_i(x,t)=\langle n_i(x,t)\rangle.
\]
于是得到格子 Boltzmann 型方程
\[
f_i(x+c_i\Delta t,t+\Delta t)-f_i(x,t)=\Omega_i(f).
\]

一个最方便的具体选择是 BGK 碰撞：
\[
\Omega_i=-\frac1\tau\left(f_i-f_i^{eq}\right),
\]
\[
f_i^{eq}=w_i\rho\left[
1+\frac{c_i\cdot u}{c_s^2}
+\frac{(c_i\cdot u)^2}{2c_s^4}
-\frac{|u|^2}{2c_s^2}
\right].
\]

这个结构已经足够“像硬球”：局部碰撞、守恒、有平衡态、有 H 定理。区别只是碰撞核从硬球积分换成了局部松弛。

---

## 3. 从结构到 Boltzmann 方程

定义经验测度
\[
\mu_t^N=\frac1N\sum_{i=1}^N\delta_{(x_i(t),v_i(t))}.
\]
在 Boltzmann–Grad 极限
\[
N\to\infty,\qquad a\to0,\qquad Na^2\to \text{常数}
\]
下，若初始状态混沌，则
\[
\mu_t^N\rightharpoonup f(t,x,v)\,dx\,dv,
\]
且 \(f\) 满足 Boltzmann 方程
\[
\partial_t f+v\cdot\nabla_x f=Q(f,f),
\]
其中硬球碰撞核为
\[
Q(f,f)=\sigma^2\int_{\mathbb R^3}\int_{S^2_+}
|(v-v_*)\cdot\omega|
\left[f(v')f(v_*')-f(v)f(v_*)\right]
\,d\omega\,dv_*.
\]
这里 \(\sigma\) 是硬球直径。

平衡态是 Maxwell 分布：
\[
f^0=\frac{\rho}{(2\pi T)^{3/2}}
\exp\left(-\frac{|v-u|^2}{2T}\right),
\]
这里取 \(m=k_B=1\)。

---

## 4. 从 Boltzmann 到 Navier–Stokes

<details>
<summary>Chapman–Enskog 展开细节</summary>

令 \(\epsilon\) 为 Knudsen 数，作展开
\[
f=f^0+\epsilon f^1+\epsilon^2 f^2+\cdots,
\]
\[
\partial_t=\epsilon\partial_{t_1}+\epsilon^2\partial_{t_2}+\cdots,
\qquad
\nabla=\epsilon\nabla_1.
\]

零阶：
\[
Q(f^0,f^0)=0.
\]

一阶：
\[
(\partial_{t_1}+v\cdot\nabla_1)f^0=2Q(f^0,f^1).
\]
定义线性化碰撞算子
\[
Lg=-2Q(f^0,g),
\]
则
\[
Lf^1=-(\partial_{t_1}+v\cdot\nabla_1)f^0.
\]
其核空间为
\[
\ker L=\operatorname{span}\{1,v,|v|^2\}.
\]
可解性条件
\[
\int \psi_a(\partial_{t_1}+v\cdot\nabla_1)f^0\,dv=0,
\qquad
\psi_a\in\{1,v,|v|^2\},
\]
给出 Euler 方程：
\[
\partial_t\rho+\nabla\cdot(\rho u)=0,
\]
\[
\partial_t(\rho u)+\nabla\cdot(\rho u\otimes u+pI)=0,
\]
\[
\partial_t E+\nabla\cdot[(E+p)u]=0.
\]

二阶修正给出
\[
\tau_{ij}=-\int c_ic_j f^1\,dv,
\qquad
q_i=\frac12\int c_i|c|^2 f^1\,dv.
\]
对硬球气体，
\[
\tau_{ij}=\mu\left(
\partial_i u_j+\partial_j u_i-\frac23\delta_{ij}\partial_k u_k
\right),
\]
\[
q_i=-\kappa\partial_i T.
\]

</details>

最终得到可压缩 Navier–Stokes 方程：
\[
\partial_t\rho+\nabla\cdot(\rho u)=0,
\]
\[
\partial_t(\rho u)+\nabla\cdot(\rho u\otimes u)
=-\nabla p+\nabla\cdot\tau,
\]
\[
\partial_t E+\nabla\cdot[(E+p)u]
=\nabla\cdot(\tau\cdot u)+\nabla\cdot(\kappa\nabla T),
\]
其中
\[
\tau=\mu\left[\nabla u+(\nabla u)^T-\frac23(\nabla\cdot u)I\right].
\]

对单原子硬球气体，Navier–Stokes 阶体积粘性为零；若有内部自由度，则还要加 \(\zeta(\nabla\cdot u)I\)。

硬球的输运系数为
\[
\mu=\frac{5}{16\sigma^2}\sqrt{\frac{m k_BT}{\pi}},
\qquad
\kappa=\frac{15}{4}\frac{k_B}{m}\mu,
\]
因此
\[
\Pr=\frac{\mu c_p}{\kappa}=\frac23.
\]

在低马赫数、近似不可压极限下，
\[
\nabla\cdot u=0,
\]
\[
\partial_t u+(u\cdot\nabla)u
=-\frac1\rho\nabla p+\nu\Delta u,
\qquad
\nu=\frac{\mu}{\rho}.
\]

---

## 5. 离散版本的 NS 极限

对格子 Boltzmann 方程
\[
f_i(x+c_i\Delta t,t+\Delta t)-f_i(x,t)
=-\frac1\tau(f_i-f_i^{eq}),
\]
Chapman–Enskog 展开给出
\[
\nu=c_s^2\left(\tau-\frac12\right)\Delta t,
\qquad
p=c_s^2\rho.
\]
因此同样得到
\[
\partial_t u+(u\cdot\nabla)u
=-\frac1\rho\nabla p+\nu\Delta u,
\qquad
\nabla\cdot u=0.
\]

要保证这一步成立，格子张量必须满足各向同性条件：
\[
\sum_i w_i c_{i\alpha}c_{i\beta}
=c_s^2\delta_{\alpha\beta},
\]
\[
\sum_i w_i c_{i\alpha}c_{i\beta}c_{i\gamma}c_{i\delta}
=c_s^4\left(
\delta_{\alpha\beta}\delta_{\gamma\delta}
+\delta_{\alpha\gamma}\delta_{\beta\delta}
+\delta_{\alpha\delta}\delta_{\beta\gamma}
\right).
\]
这就是 D2Q9、D3Q19 等速度集要满足的条件。

---

## 6. 设计检查清单

要让一个“硬球型结构”最终给出 NS，检查：

1. **局部碰撞**：碰撞只依赖有限邻域。
2. **守恒**：质量、动量、能量严格守恒。
3. **各向同性**：碰撞算子或格子张量足够对称。
4. **H 定理**：存在平衡态且熵产生非负。
5. **尺度分离**：\(Kn\ll1\)，Chapman–Enskog 截断有效。
6. **热模型**：若同时要正确的 \(\mu\) 和 \(\kappa\)，不能只用单弛豫 BGK，因为 BGK 给出 \(\Pr=1\)；硬球给出 \(\Pr=2/3\)，需要多弛豫/MRT 或真实硬球碰撞算子。
7. **不可压极限**：低马赫数、密度涨落小。

---

## 7. 结论

可以这样概括：

> **硬球型结构 = 局部守恒碰撞 + 可逆性 + 各向同性 + H 定理 + 尺度分离。**  
> 硬球只是取碰撞核
> \[
> B\propto |(v-v_*)\cdot\omega|
> \]
> 的特例。任何满足上述结构律的模型，在 Boltzmann–Grad 极限和 Chapman–Enskog 二阶截断下，都会给出可压缩 NS；在低马赫数下给出不可压缩 NS。

因此，“类似硬球”不一定要真的画球。你可以用格子占据数、局部碰撞表、甚至 BGK 松弛来替代硬球碰撞；只要保留守恒、各向同性和 H 定理，NS 方程仍会作为其流体力学极限出现。

---
## 🧑‍💻 User

重新思考粗粒化的方式，并且计算机友好，可以算稀薄气体

---
## 😀 默认助手 (7)

上一版把粗粒化理解为“取矩 + Chapman–Enskog 截断”，这在 \(Kn\ll1\) 时成立，但对稀薄气体不成立：非平衡部分不是小量，不能把速度分布强行投影到局部 Maxwell 分布。要算稀薄气体，粗粒化必须改成**保留非平衡部分、按局部 Kn 自适应选择表示**。

核心改动只有一句：

\[
f=\underbrace{f^{eq}(\rho,u,T)}_{\text{永远可以粗粒化}}+\underbrace{g}_{\text{非平衡部分，不能随便丢}}.
\]

NS 方程相当于强制 \(g\approx g_{NS}(\nabla\rho,\nabla u,\nabla T)\)；稀薄气体必须让 \(g\) 自己演化。

---

## 1. 新的粗粒化原则

### 1.1 旧路线的问题

\[
f \xrightarrow{\text{取矩}} (\rho,u,T)
\xrightarrow{\text{Chapman–Enskog}} NS.
\]

这只在局部平衡附近成立。稀薄气体中 \(g=f-f^{eq}\) 与 \(f^{eq}\) 同阶，丢掉 \(g\) 就等于丢掉物理。

### 1.2 新路线：micro–macro 分解

把状态写成
\[
f=f^{eq}(M)+g,\qquad M=(\rho,u,T).
\]
- \(M\)：宏观量，始终粗粒化；
- \(g\)：非平衡缺陷，按局部 \(Kn\) 选择表示：
  - 连续区：\(g\approx g_{NS}\)，甚至 \(g=0\)；
  - 过渡区：\(g\) 用低阶矩或 ES-BGK 表示；
  - 稀薄区：\(g\) 用粒子或离散速度完整保留。

这就是“自适应相空间粗粒化”：**粗粒化不是全局一次性的投影，而是局部的、可逆的、带误差控制的限制/延拓。**

---

## 2. 结构定义

定义计算结构
\[
\mathfrak K=(\mathcal P,\mathcal G,\mathcal V,\mathcal C,\mathcal S,\mathcal A),
\]
其中：

| 分量 | 含义 |
|---|---|
| \(\mathcal P\) | 粒子或速度分布表示 |
| \(\mathcal G\) | 空间网格/自适应子胞 |
| \(\mathcal V\) | 速度空间表示：粒子、离散速度、矩 |
| \(\mathcal C\) | 局部碰撞算子 |
| \(\mathcal S\) | 采样/取矩算子 |
| \(\mathcal A\) | 自适应与重采样算子 |

### 2.1 数据布局

计算机友好版本建议用 SoA：

```cpp
struct Particles {
    std::vector<double> x, y, z;   // 位置
    std::vector<double> vx, vy, vz; // 速度
    std::vector<double> w;          // 统计权重
    std::vector<int> cell;          // 所属网格
};

struct Cell {
    double V;                       // 体积
    double rho, u[3], T;            // 宏观量
    double Kn_local;
    int model;                      // NS / ES_BGK / DSMC / DVM
    std::vector<int> particle_ids;
};
```

这种布局可以直接上 GPU：输运、分箱、碰撞、采样都是局部操作。

---

## 3. 操作定义

### 3.1 主循环

```cpp
for (int step = 0; step < nsteps; ++step) {
    advect(particles, dt);          // 1. 弹道输运
    bin_particles(particles, cells); // 2. 计数排序分箱
    parallel_for (cell : cells) {   // 3. 局部演化
        if (cell.model == NS) {
            solve_NS(cell, dt);
        } else if (cell.model == ES_BGK) {
            relax_ES_BGK(cell, dt);
        } else if (cell.model == DSMC) {
            dsmc_collide(cell, dt);
        } else if (cell.model == DVM) {
            dvm_collide(cell, dt);
        }
    }
    parallel_for (cell : cells) {   // 4. 取矩
        cell.macros = sample_moments(cell);
    }
    adapt(cells);                   // 5. 自适应
    resample(particles, cells);     // 6. 重采样
    couple_interfaces(cells);       // 7. 连续/稀薄耦合
}
```

### 3.2 各操作的具体含义

**Advect：**
\[
x_i \leftarrow x_i+v_i\Delta t.
\]
粒子版本是 \(O(N)\)；连续区版本用有限体积通量。

**Bin：**
用计数排序把粒子分到网格，\(O(N)\)，避免每步做空间搜索。

**Collide：**
- DSMC：局部随机碰撞；
- ES-BGK：向修正 Maxwell 分布松弛；
- DVM：在速度节点上做确定性碰撞；
- NS：不碰粒子，直接解宏观方程。

**Sample：**
\[
\rho=\sum_{i\in c} m_iw_i,\qquad
\rho u=\sum_{i\in c} m_iw_iv_i,
\]
\[
P_{ij}=\sum_{i\in c} m_iw_i(v_i-u)_i(v_i-u)_j,
\qquad
q_i=\sum_{i\in c}\frac12m_iw_i|v_i-u|^2(v_i-u)_i.
\]

**Resample：**
控制每个网格的粒子数和权重方差。低权重粒子合并，高权重粒子分裂。

**Adapt：**
根据局部 \(Kn\)、梯度、粒子数决定模型层级、网格尺寸和速度表示。

---

## 4. 碰撞算子怎么选

### 4.1 稀薄区：DSMC

DSMC 就是字面意义上的硬球模型，最适合高 \(Kn\)。

NTC 碰撞数：
\[
N_{\rm coll}
=\frac12 N\bar N
\frac{\sigma_T c_{r,\max}\Delta t}{V},
\]
其中 \(N\) 是网格内粒子数，\(\bar N\) 是平均粒子数，\(c_{r,\max}\) 是最大相对速度，\(\sigma_T\) 是碰撞截面。候选对以概率 \(c_r/c_{r,\max}\) 接受。

硬球截面：
\[
\sigma_T=\pi d^2.
\]
高温稀薄气体建议用 VHS/VSS：
\[
\sigma_T\propto c_r^{-2\omega}.
\]

### 4.2 过渡区：ES-BGK / Shakhov

BGK：
\[
\partial_t f+v\cdot\nabla f=\nu(f^{eq}-f),
\qquad
\nu=\frac{p}{\mu}.
\]
BGK 的 Prandtl 数是 \(1\)，硬球是 \(2/3\)，所以热问题要用 ES-BGK 或 Shakhov 修正。

ES-BGK 用修正温度张量 \(T_{ij}\) 构造高斯分布，使应力松弛和热流松弛的速率不同，从而匹配正确的 \(\Pr\)。

### 4.3 连续区：NS

当 \(Kn_{\rm local}\) 很小时，直接解
\[
\partial_t\rho+\nabla\cdot(\rho u)=0,
\]
\[
\partial_t(\rho u)+\nabla\cdot(\rho u\otimes u)
=-\nabla p+\nabla\cdot\tau,
\]
\[
\partial_t E+\nabla\cdot[(E+p)u]
=\nabla\cdot(\tau\cdot u)+\nabla\cdot(\kappa\nabla T).
\]

### 4.4 统一方法：DUGKS / UGKS

如果不想做随机粒子，可以用离散速度 + 有限体积的 DUGKS/UGKS。它们直接在速度网格上解 Boltzmann/BGK 方程，从连续区到稀薄区都有效，且没有统计噪声。

代价是速度空间维数高、内存大；可以用自适应速度网格缓解。

---

## 5. 自适应准则

局部 Knudsen 数：
\[
Kn_{\rm local}=\frac{\lambda}{L_{\rm local}},
\]
其中
\[
\lambda=\frac{\mu}{p}\sqrt{\frac{\pi}{2RT}}
\]
是平均自由程，\(L_{\rm local}\) 可以取
\[
L_{\rm local}
=\min\left(
\frac{\rho}{|\nabla\rho|},
\frac{|u|}{|\nabla u|},
\frac{T}{|\nabla T|}
\right).
\]

一个可直接用的切换策略：

| 局部 \(Kn\) | 表示 |
|---|---|
| \(Kn<10^{-2}\) | NS |
| \(10^{-2}<Kn<0.1\) | ES-BGK / DUGKS |
| \(Kn>0.1\) | DSMC |

实际实现时不要硬切，留一个重叠缓冲区，用通量耦合，避免界面反射。

---

## 6. 误差控制

计算机友好不等于无脑算，必须有误差指标：

1. **统计误差**：
\[
\epsilon_{\rm stat}\sim\frac{1}{\sqrt{N_c}},
\]
\(N_c\) 是网格内有效粒子数。低马赫数稀薄气体要控制权重方差，必要时用低方差 DSMC 或 deviational 方法。

2. **离散误差**：
DSMC 要求
\[
\Delta x\lesssim\lambda,\qquad
\Delta t\lesssim\tau_c,
\]
其中 \(\tau_c\) 是平均碰撞时间。可用自适应子胞避免全局加密。

3. **模型误差**：
用 \(Kn_{\rm local}\) 和梯度判据估计；在模型界面上做通量守恒检查。

4. **速度表示误差**：
DVM 的速度网格分辨率要能分辨 \(f\) 的宽度；粒子表示则用重采样控制噪声。

---

## 7. NS 怎么“最后一样获得”

在这个结构里，NS 不是全局假设，而是局部极限：

- 在连续区，令 \(g=0\)，对 Boltzmann 方程做 Chapman–Enskog 展开，得到 NS；
- 在稀薄区，保留 \(g\)，解 kinetic 方程；
- 当 \(Kn\to0\) 时，\(g=O(Kn)\to0\)，整个区域自动退化为 NS。

也就是说：

\[
\boxed{
\text{NS 是自适应 kinetic 结构在 }Kn\to0\text{ 时的局部限制，}
}
\]

而不是一开始就强加的闭包。

---

## 8. 推荐实现路线

### 路线 A：字面硬球，先能算稀薄气体

1. DSMC + 自适应子胞；
2. VHS/VSS 碰撞模型；
3. 加权粒子 + 重采样；
4. 计数排序分箱；
5. 稳态用时间平均，瞬态用 Strang splitting。

这条路线最接近“硬球结构”，也最容易验证。

### 路线 B：低噪声、多尺度

1. DUGKS/UGKS + 自适应速度网格；
2. 连续区用 NS 有限体积；
3. 界面通量耦合；
4. 用 \(Kn_{\rm local}\) 切换模型。

这条路线更适合低马赫数微流动和稳态问题。

### 路线 C：混合 DSMC–NS

- 稀薄区：粒子 DSMC；
- 连续区：NS 有限体积；
- 重叠区：DSMC 给 NS 提供应力/热流通量，NS 给 DSMC 提供边界分布；
- 用粒子数、\(Kn\)、梯度三重判据自适应。

---

## 9. 总结

重新设计后的粗粒化可以概括为：

\[
\boxed{
\text{不要粗粒化掉整个分布函数，只粗粒化平衡部分；非平衡部分按局部 }Kn\text{ 自适应表示。}
}
\]

结构上：

\[
\mathfrak K=(\mathcal P,\mathcal G,\mathcal V,\mathcal C,\mathcal S,\mathcal A),
\]

操作上：

\[
\text{Advect} \to \text{Bin} \to \text{Collide} \to \text{Sample} \to \text{Adapt} \to \text{Resample}.
\]

这样得到的代码：

- 对稀薄气体：DSMC / DVM / ES-BGK 直接解 kinetic 方程；
- 对连续区：自动退化为 NS；
- 对多尺度：用 \(Kn_{\rm local}\) 做局部模型切换；
- 对计算机：操作局部、并行、\(O(N)\)、GPU 友好。
