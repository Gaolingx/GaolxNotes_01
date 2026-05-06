# FIR和IIR滤波器方程及回声设计

### 🧑‍💻 User

4. (a) Give the mathematical equations for FIR and IIR filters. Use x(n) as an input sequence and y(n) as an output sequence to describe their relationship in terms of sequence equations for both filters.
(b) Assume x(n) is sampled from an analog signal with sampling rate 20kHz, write down the FIR for producing the signal with an echo with amplitude reduced by 50% and delayed by 0.05 seconds.

---

### 🤖 Assistant

Here are the mathematical equations and explanations for both parts of your question.

### (a) Mathematical Equations for FIR and IIR Filters

Let $x(n)$ be the input sequence and $y(n)$ be the output sequence. The relationship between the input and output in the time domain is described by linear constant-coefficient difference equations.

#### **1. FIR (Finite Impulse Response) Filter**
An FIR filter only uses current and past input values to calculate the output. It does not use feedback.

The general sequence equation is:
$$y(n) = \sum_{k=0}^{M-1} b_k x(n-k)$$

**Where:**
*   $M$ is the length of the filter (number of coefficients).
*   $b_k$ represents the feedforward filter coefficients.
*   $x(n-k)$ represents the current and past input samples.

#### **2. IIR (Infinite Impulse Response) Filter**
An IIR filter uses current and past input values, as well as past output values (feedback), to calculate the current output. 

The general sequence equation is:
$$y(n) = \sum_{k=0}^{M} b_k x(n-k) - \sum_{k=1}^{N} a_k y(n-k)$$

*(Note: Sometimes this is written with a `+` for the feedback terms depending on convention, but the standard difference equation defines $a_0 y(n) + \sum_{k=1}^{N} a_k y(n-k) = \sum_{k=0}^{M} b_k x(n-k)$ where $a_0 = 1$.)*

**Where:**
*   $M$ is the feedforward filter order.
*   $N$ is the feedback filter order.
*   $b_k$ represents the feedforward coefficients.
*   $a_k$ represents the feedback coefficients.
*   $y(n-k)$ represents the past output samples (feedback).

---

### (b) FIR Equation for an Echo Effect

To write the sequence equation for this specific FIR filter, we first need to determine the delay in terms of discrete samples rather than seconds.

**1. Calculate the delay in samples ($D$):**
We are given:
*   Sampling rate ($f_s$) = $20 \text{ kHz} = 20000 \text{ samples/second}$
*   Time delay ($t_d$) = $0.05 \text{ seconds}$

$$D = t_d \times f_s$$
$$D = 0.05 \times 20000 = 1000 \text{ samples}$$

**2. Formulate the FIR sequence equation:**
The output $y(n)$ should consist of the original signal $x(n)$ plus the echo. The echo is the input signal delayed by $1000$ samples, $x(n-1000)$, and reduced in amplitude by 50% ($0.5$).

The resulting FIR equation is:
$$y(n) = x(n) + 0.5 x(n-1000)$$

<details>
<summary><b>Click here to see the filter coefficients for this equation</b></summary>
If we express this specific equation using the standard FIR summation formula $y(n) = \sum_{k=0}^{1000} b_k x(n-k)$, the filter coefficients ($b_k$) would be:
<ul>
<li>$b_0 = 1$ (The original signal)</li>
<li>$b_{1000} = 0.5$ (The echo)</li>
<li>$b_k = 0$ for all other values of $k$ between $1$ and $999$</li>
</ul>
</details>