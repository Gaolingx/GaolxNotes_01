# PSNR SI-SNR SNR区别

### 🧑‍💻 User

psnr si-snr snr metrics有什么区别？

---

### 🤖 Assistant

Let’s break down the differences between **PSNR**, **SNR**, and **SI‑SNR** (also called SI‑SDR). They all measure how similar an estimated signal $\hat{s}$ is to a clean reference $s$, but they differ in **scale sensitivity**, **normalisation**, and typical **application**.

---

## 1. SNR – Signal-to-Noise Ratio

**Definition**
Compares the power of the reference signal to the power of the error (difference):
$$
\text{SNR} = 10 \log_{10} \frac{\|s\|^2}{\|s - \hat{s}\|^2} \quad [\text{dB}]
$$

**Key properties**
- **Scale‑sensitive** – if $\hat{s} = c \cdot s$, the error is non‑zero and SNR drops (even for a perfect match up to gain).
- **Time‑alignment** required – a small time shift will dramatically reduce SNR.
- **Unbounded** – perfect reconstruction ($\hat{s} = s$) gives $\infty$ dB.
- Used as a general quality measure in many signal‑processing tasks.

---

## 2. PSNR – Peak Signal-to-Noise Ratio

**Definition**
Normalises the error by the **maximum possible signal amplitude** ($\text{MAX}$):
$$
\text{PSNR} = 10 \log_{10} \frac{\text{MAX}^2}{\frac{1}{N}\sum_{n} (s[n] - \hat{s}[n])^2}
         = 20 \log_{10} \frac{\text{MAX}}{\sqrt{\text{MSE}}} \quad [\text{dB}]
$$

- For audio, $\text{MAX}$ is usually **1** (normalised float signals) or $2^{15}\!-\!1$ (16‑bit PCM).
- Equivalent to SNR if the signal $s$ has constant amplitude $\text{MAX}$. Otherwise, PSNR ignores the actual signal power.

**Key properties**
- **Scale‑sensitive** via the fixed peak – scaling $\hat{s}$ changes the MSE.
- **No dependency on signal power** – a quiet signal can give a lower SNR but a higher PSNR.
- Standard metric in **image/video** coding; sometimes used in audio when the dynamic range is bounded (e.g., speech coding with a fixed peak).
- Also time‑alignment dependent.

---

## 3. SI‑SNR (Scale‑Invariant Signal‑to‑Noise Ratio)

**Definition** (also known as SI‑SDR)
Projects $\hat{s}$ onto $s$ to find the **optimal scaling factor**, then measures the error:
$$
s_{\text{target}} = \frac{\langle \hat{s}, s \rangle}{\|s\|^2}\,s, \qquad
e_{\text{noise}} = \hat{s} - s_{\text{target}}
$$
$$
\text{SI‑SNR} = 10 \log_{10} \frac{\|s_{\text{target}}\|^2}{\|e_{\text{noise}}\|^2} \quad [\text{dB}]
$$

**Key properties**
- **Scale‑invariant** – any scaling of $\hat{s}$ (e.g., $c\hat{s}$) does **not** change SI‑SNR, because the gain is absorbed by the projection.
- Still requires **time alignment**; a shift will degrade the score.
- Perfect reconstruction ($\hat{s} = c s$) gives $\infty$ dB.
- Dominant metric in **speech separation / enhancement** (e.g., TasNet, Conv‑TasNet) as it focuses on waveform integrity regardless of level.

---

## Summary Table

| Metric | Formula essence | Sensitive to scaling | Sensitive to time shift | Typical use |
|--------|----------------|----------------------|------------------------|-------------|
| **SNR** | $\displaystyle 10\log_{10}\frac{\|s\|^2}{\|s-\hat{s}\|^2}$ | Yes | Yes | General signal quality |
| **PSNR** | $\displaystyle 10\log_{10}\frac{\text{MAX}^2}{\text{MSE}}$ | Yes (but via fixed peak) | Yes | Image/video coding, sometimes audio |
| **SI‑SNR** | $\displaystyle 10\log_{10}\frac{\|s_{\text{target}}\|^2}{\|e_{\text{noise}}\|^2}$ | **No** | Yes | Source separation, waveform enhancement |

---

## When to choose which

- **SI‑SNR** is preferred in **source separation** because you care about the signal shape, not its overall level. A separated source that is too quiet can simply be rescaled – the distortion should not be penalised.
- **SNR** is natural when you know the exact reference level matters (e.g., measuring noise in a calibrated recording chain).
- **PSNR** is less common for audio but appears when you want a metric tied to the full‑scale range (like comparing codecs that clip at ±1). For speech or music with varying loudness, SNR or SI‑SNR usually correlate better with perception.

All three are **objective, sample‑level** metrics and do not directly model human perception (unlike PESQ, STOI, etc.). They are computationally cheap and useful for training and quick evaluation.