# Glitch Token与外部安全护栏机制及校验方法详解

## 🧑‍💻 User

llm破限常见手段之Glitch token就是强制改写概率（补全api），所以才会有external safeguard rail，专门校验这个，具体是怎么做的？

---
## 😀 默认助手 (7)

# Glitch Token 与外部安全护栏：机制与校验方法

先厘清一个常被混用的点，因为“怎么校验”取决于“它到底改了什么”。

## 一、概念校准：glitch token 到底篡改了什么

你的说法“glitch token 就是强制改写概率”方向对，但需要拆成两类，因为护栏对二者的校验逻辑不同：

| 类型 | 本质 | 对概率分布的作用 | 典型代表 |
|---|---|---|---|
| **Glitch / undertrained token** | 词表里有该 token，但预训练语料中出现次数≈0，embedding 与输出投影行几乎没被训练 | 把隐藏态推离训练分布（OOD），使 $P(x_t\mid x_{<t})$ **退化/塌缩**：熵要么爆炸要么塌到单一怪 token 上 | ` SolidGoldMagikarp`、` davidjl`、` petertodd`、` edRaccoon`、` ScaledLion` 等 |
| **Adversarial suffix (GCG)** | 用梯度搜索**优化**一段后缀，使目标续写的概率最大化 | **真正的“强制改写概率”**：$\min_{s} -\log p(\text{target}\mid x\oplus s)$ | Zou et al. 2023 的 universal suffix |

所以更准确地说：glitch token 并非去“改写”某个具体 token 的概率；它把模型推进一个训练时没见过的表征区，导致**下一 token 分布失去语义结构**；而补全 API 直接暴露原始 next-token 采样，没有 chat template 和指令层去吸收这种扰动，因此表现得最赤裸。真正“定向改写概率”的是 GCG 那类后缀攻击。

**这也是为什么外部护栏不能只看文本**：` davidjl` 的字符串形态看起来像普通用户名，文本分类器毫无反应；只有回到 **token id、语料频率、embedding 与 logits** 层面才能看到异常信号。

---

## 二、为什么必须“外置”护栏

模型自身无法对自己做概率分布的异常校验；一旦它被推入 OOD 区，它连“我此刻不正常”都判断不了。所以护栏必须放在**推理栈外部或旁路**，并且要能拿到文本层拿不到的信号：

- 原始字节与规范化后的字节
- **token id 序列**（而非字符串）
- **logits、熵、top-k 质量**
- （可选的）隐藏态

核心原则：**护栏观测的抽象层要比攻击者操纵的抽象层更低或等深**。攻击者在 token 层下毒，护栏就必须在 token 层验毒。

---

## 三、具体校验手段（分层）

<details>
<summary><b>A. 词表 / 分词层：规范化与“非幂等”检测</b></summary>

1. **Unicode 规范化**：NFC/NFKC，剥离零宽字符（U+200B/200C/200D/FEFF）、双向控制符、未分配码位、组合字符堆叠。Glitch 攻击常借“看起来一样但编码不同”绕过。
2. **分词幂等性检测**：`tokenize(detokenize(ids)) != ids` 即为可疑。正常文本重分词应稳定；出现不幂等往往意味着 tokenizer 边界被利用（token boundary attack / 多字节 merge 伪影）。
3. **词表合法性**：若服务端能把词表裁剪到“活跃词表”（见下文 G），任何落在裁剪外的 id 直接判非法。
</details>

<details>
<summary><b>B. 语料频率层：glitch token 最本质的指纹</b></summary>

Glitch token 的定义性特征就是**训练语料中频次极低的 token**。护栏维护一张按语料统计的频率表：

$$f(t)=\frac{c(t)}{N},\qquad \text{flag if } f(t)<\tau_f$$

工程上 $\tau_f$ 常取 $10^{-8}\sim10^{-7}$ 量级。但要注意 **SolidGoldMagikarp 型签名**：它在总量上不低，却几乎只来自**单一文档/crawl dump**（一篇 Reddit 帖被重复灌入），因此还要看**文档级集中度**：

$$\text{concentration}(t)=\frac{\max_d c_d(t)}{c(t)} \approx 1 \Rightarrow \text{可疑}$$

即“总量小”或“总量正常但集中于单一来源”都要进黑名单。这类清单是模型/词表特定的，换模型即失效，但**发现方法**可复用。
</details>

<details>
<summary><b>C. 嵌入 / OOD 层：在表征空间里验毒</b></summary>

1. **Mahalanobis 距离**：用训练分布估计 $\mu,\Sigma$，对 token embedding 或某层隐藏态算

$$D(x)=\sqrt{(x-\mu)^\top\Sigma^{-1}(x-\mu)}$$

超阈值即 OOD。Glitch token 的 embedding 常表现为“邻居簇稀疏/余弦近邻诡异”。
2. **kNN 密度**：在 embedding 空间做近邻密度估计，低密度区 token 标记。
3. **逐 token loss 剖面**：前向一遍，看每个位置的 loss 曲线；glitch token 处出现异常尖峰或异常低谷。
</details>

<details>
<summary><b>D. 概率 / 熵层：你说的“校验改概率”就落在这里</b></summary>

护栏（若在推理栈内可拿到 logits，或通过 logprobs 接口）监控下一 token 分布的形状：

$$H_t=-\sum_{v}p(v\mid x_{<t})\log p(v\mid x_{<t}),\qquad p_{\max}=\max_v p(v\mid x_{<t})$$

- **熵塌缩**：$H_t$ 骤降、$p_{\max}\to 1$ 且 top-1 是远离语义的 token，说明模型被吸引子吸住。
- **熵爆炸**：$H_t$ 远高于同上下文基线，说明分布失去结构。
- **glitch 概率穿透**：直接算 $P(\text{glitch}\mid x_{<t})$。若它显著高于任何“上下文合理 token”，直接拦截。
- **重复/吸引子检测**：统计 n-gram 重复率；某怪 token 被反复吐出，是补全被劫持的强信号。
- **参考模型困惑度对比**：用另一模型（或同模型不同温度）算输入困惑度，异常高/异常低都告警。

> 注意：这类检查对 **GCG 后缀**同样有效，因为后缀的目标就是让 $p_{\max}$ 偏向 affirmative 前缀，熵形貌会异常；但它对“频率黑名单”无效，因为 GCG 用的是**训练充分的普通 token**。
</details>

<details>
<summary><b>E. 表征 / 注意力层：浅层探针与指令层级监控</b></summary>

1. **线性安全探针**：在中间层隐藏态上训练“越狱/攻击进行中”分类器（representation engineering / Circuit Breakers 思路）。它对未见过的攻击有泛化力。
2. **注意力坍缩监控**：比较“系统提示 token 的注意力质量”相对基线的变化；后缀/glitch 生效时，对系统提示的注意力常显著下降。
3. **系统提示 canary**：系统提示里埋随机串，若输出回显该串或缺省遵循 schema，判定被劫持。
</details>

<details>
<summary><b>F. 输出层：最后一道确定性闸门</b></summary>

1. **回显 glitch token 即拦**：输出里出现黑名单 token 的 id，几乎可断定补全被污染。
2. **安全分类器**：Llama Guard / WildGuard / 云厂商内容安全 API，对 completion 做策略判定。
3. **形状校验**：是否满足期望 schema/JSON、是否 coherent、是否连续重复同 token 多次。
4. **多采样一致性**：$k$ 次采样若发散剧烈（低一致率），判为不稳定并重生成。
</details>

<details>
<summary><b>G. 解码约束层：最“硬”的防线（往往比检测更有效）</b></summary>

检测是“事后抓”，约束是“根本不让它发生”：

- **受限解码 / 语法约束**：用 JSON schema、GBNF、FSM 约束解码（outlines、guidance）。Glitch token 通常不在合法文法内，直接被 mask 掉；这一招对工具调用/结构化输出场景近乎根治。
- **logit masking**：服务端对已知 glitch token id 的 logit 置 $-\infty$；或做“核心词表 top-k 截断”，只允许训练充分的 token。
- **logit floor/ceiling、min-p / typical sampling**：压制长尾怪 token 的采样路径。
- **Tokenizer healing**：生成前对结尾 token 做重分词修复，避免尾 token 伪影。
</details>

---

## 四、一条可落地的校验流水线

```python
def guard_input(prompt_str):
    # 0) 规范化 + 字符层
    s = nfc_normalize(strip_zero_width(prompt_str))
    if has_unassigned_or_bidi(s): return BLOCK("char-anomaly")

    # 1) token 层：分词幂等
    ids = tokenizer(s)
    if tokenizer(tokenizer.decode(ids)) != ids: return BLOCK("non-idempotent")

    # 2) 词表频率层（双信号）
    for t in ids:
        if freq[t] < TAU_F:                       return BLOCK("rare-token")
        if doc_concentration[t] > 0.8 and freq[t] < TAU_LOW: return BLOCK("dump-token")
        if t in GLITCH_BLOCKLIST:                 return BLOCK("known-glitch")

    # 3) 嵌入 OOD
    for t in ids:
        if mahalanobis(emb[t]) > TAU_OOD:         return BLOCK("embed-ood")

    # 4) 概率/熵层（若可拿 logits/logprobs）
    logits = model.forward(s)                     # 或 logprobs 接口
    H, pmax = entropy(logits), top1(logits)
    if pmax > 0.98 and top1_id in GLITCH_BLOCKLIST: return BLOCK("attractor")
    if H > H_BASE * 3:                            return FLAG("entropy-explosion")
    if P(logits, GLITCH_IDS) > P(logits, plausible): return BLOCK("glitch-dominates")

    # 5) 浅层探针
    if probe(hidden_states(s)) > TAU_PROBE:       return FLAG("jailbreak-probe")

    # 6) 输出层（生成后）
    out = generate(s)
    if any(t in GLITCH_BLOCKLIST for t in tokenizer(out)): return REGENERATE("echo-glitch")
    if not safe_classifier(out):                  return BLOCK("policy")
    if not schema_ok(out):                        return REGENERATE("schema")

    return ALLOW
```

关键工程点：**第 2、3、4 步必须拿 token id / logits**，不能只拿字符串；**黑名单要闭环**——每次拦截样本自动回灌语料频率统计与 blocklist，形成“发现新 glitch token → 拦截 → 扩充”的反馈。

---

## 五、局限与对抗演化

<details>
<summary><b>为什么这套不是万能的</b></summary>

1. **模型/词表特定**：glitch 清单换模型即失效；护栏必须按模型重算频率与 embedding 统计。
2. **频率检测挡不住 GCG**：普通 token 组成的对抗后缀频率正常，必须靠熵/困惑度/探针/受限解码来挡；这也是为什么“只想靠 glitch 黑名单”的护栏会被同类攻击绕过。
3. **受限解码的代价**：语法约束只适用于结构化输出场景，开放对话里语法过宽，压制效果下降。
4. **假阳性**：罕见专有名词、代码、非英语文本天然低频，阈值过严会误杀，需要按语言/领域分层设阈。
5. **厂商侧根治**：真正长期的修法是**继续预训练给低频 token 加热**、**裁词表**、**换更干净的分词器**（tiktoken o200k 一类），以及服务端 logit bias/banned-token 列表——这属于模型供给方，而非客户侧护栏能做到的。

**一句话总结**：外部 safeguard rail 校验 glitch token 的本质，是把判定从“文本语义”下沉到“**token 频率 + embedding OOD + 分布熵形貌**”这三个攻击者无法在字符串层面伪装的信号上；而最硬的防线是**受限解码与 logit 屏蔽**，让怪 token 根本没有被采样的机会。
</details>

如果你想深入某一块（比如熵检测的具体阈值标定、受限解码的 FSM 实现、或 GCG 后缀的检测特征），我可以展开。
