# LoRA权重合并脚本

### 🧑‍💻 User

编写一份lora合并脚本，支持pth/safetensors输入，扫描是否包含lora ab等key，如果有则合并他们，然后走hf接口导出safetensors

---

### 🤖 Assistant

下面是一份完整的、生产可用的 LoRA 合并脚本，支持 `.pth` / `.safetensors` 输入，自动扫描 `lora_A` / `lora_B` 键并合并，通过 HF 的 `safetensors` 接口导出。

```python
#!/usr/bin/env python3
"""
LoRA Merge Script — 扫描并合并 lora_A / lora_B，导出 safetensors

支持:
  - 输入: .pth / .pt / .ckpt / .bin / .safetensors
  - 自动识别多种 LoRA key 命名约定 (PEFT / Kohya / diffusers / 自定义)
  - delta = lora_B @ lora_A  (自动推断 matmul 方向)
  - alpha / rank 缩放:  delta *= alpha / rank
  - 可选合并到 base model
  - 导出: safetensors (HF 标准格式)

用法:
  python lora_merge.py -i lora.safetensors -o merged.safetensors
  python lora_merge.py -i lora.pth       -o merged.safetensors --alpha 16
  python lora_merge.py -i lora.pth       -o merged.safetensors --base ./llama-7b
  python lora_merge.py -i lora.pth       -o merged.safetensors --base meta-llama/Llama-2-7b-hf --dtype bf16
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
from safetensors.torch import load_file as load_safetensors
from safetensors.torch import save_file as save_safetensors


# ═══════════════════════════════════════════════════════════════
# LoRA key 检测
# ═══════════════════════════════════════════════════════════════

# 匹配 lora_A 的正则集合 (支持多种前缀/分隔符)
LORA_A_REGEX = re.compile(
    r"(?:^|\.|_)lora_A(?:\.|_|$)",
    re.IGNORECASE,
)

# 匹配 lora_B 的正则集合
LORA_B_REGEX = re.compile(
    r"(?:^|\.|_)lora_B(?:\.|_|$)",
    re.IGNORECASE,
)


def is_lora_a(key: str) -> bool:
    return bool(LORA_A_REGEX.search(key))


def is_lora_b(key: str) -> bool:
    return bool(LORA_B_REGEX.search(key))


def extract_base_name(key: str) -> str:
    """
    从 lora key 提取"基名"用于配对联接 A/B。

    例子:
      "model.layers.0.self_attn.q_proj.lora_A.weight"  → "model.layers.0.self_attn.q_proj"
      "lora_te.text_model.encoder.layers.0.mlp.fc1.lora_A.weight" → "lora_te.text_model.encoder.layers.0.mlp.fc1"
    """
    # 去掉 lora_A / lora_B 及其后缀
    key = re.sub(r"[._]?lora_[AB][._]\w+$", "", key)   # .lora_A.weight
    key = re.sub(r"[._]?lora_[AB]$", "", key)           # .lora_A 结尾
    return key


# ═══════════════════════════════════════════════════════════════
# 加载
# ═══════════════════════════════════════════════════════════════

def load_weights(path: str) -> Dict[str, torch.Tensor]:
    """统一加载 .pth / .safetensors"""
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix in (".safetensors", ".sft"):
        return dict(load_safetensors(str(path)))

    if suffix in (".pth", ".pt", ".bin", ".ckpt"):
        obj = torch.load(str(path), map_location="cpu", weights_only=True)
        if isinstance(obj, dict):
            # 兼容嵌套 state_dict
            for candidate_key in ("state_dict", "model", "weights"):
                if candidate_key in obj and isinstance(obj[candidate_key], dict):
                    return dict(obj[candidate_key])
            return dict(obj)
        raise TypeError(f"Expected a dict, got {type(obj)} from {path}")

    raise ValueError(f"Unsupported format: {suffix}")


def load_base_model(model_path: str, verbose: bool = False) -> Dict[str, torch.Tensor]:
    """加载 base model (本地目录 或 HF repo id)"""
    local = Path(model_path)

    # ── 本地目录 ──
    if local.is_dir():
        weights: Dict[str, torch.Tensor] = {}
        # 优先 safetensors
        st_files = sorted(local.glob("*.safetensors"))
        bin_files = sorted(local.glob("*.bin")) if not st_files else []

        if st_files:
            for f in st_files:
                if verbose:
                    print(f"     loading {f.name} ...")
                weights.update(load_safetensors(str(f)))
        elif bin_files:
            for f in bin_files:
                if verbose:
                    print(f"     loading {f.name} ...")
                w = torch.load(str(f), map_location="cpu", weights_only=True)
                weights.update(w)
        else:
            # 尝试 index 文件
            idx = local / "pytorch_model.bin.index.json"
            if idx.exists():
                with open(idx) as fh:
                    index = json.load(fh)
                for fname in sorted(set(index.get("weight_map", {}).values())):
                    fp = local / fname
                    if fp.exists():
                        if verbose:
                            print(f"     loading {fname} ...")
                        weights.update(torch.load(str(fp), map_location="cpu", weights_only=True))
            else:
                raise FileNotFoundError(f"No weight files found in {model_path}")
        return weights

    # ── HuggingFace repo ──
    try:
        from transformers import AutoModel
    except ImportError:
        raise ImportError("需要 transformers 来从 HF Hub 加载。pip install transformers")

    print(f"     downloading '{model_path}' from HuggingFace Hub ...")
    model = AutoModel.from_pretrained(model_path, torch_dtype=torch.float32)
    state = model.state_dict()
    del model
    return dict(state)


# ═══════════════════════════════════════════════════════════════
# Delta 计算 (核心)
# ═══════════════════════════════════════════════════════════════

def compute_delta(
    a: torch.Tensor,
    b: torch.Tensor,
    a_key: str,
    b_key: str,
) -> torch.Tensor:
    """
    计算 delta = lora_B @ lora_A。

    自动推断 matmul 方向，支持:
      - PEFT / HF:   A=[r, in],  B=[out, r]  →  B @ A = [out, in]   ✓
      - Kohya:       A=[in, r],  B=[out, r]  →  B @ A^T = [out, in]
      - 其他变体:    A=[r, in],  B=[r, out]  →  B^T @ A = [out, in]
                     A=[in, r],  B=[r, out]  →  (A @ B)^T = [out, in]

    返回 shape [out_features, in_features]。
    """
    sa, sb = a.shape, b.shape

    # 情况 1: PEFT 标准 — A=[r, in], B=[out, r] → B@A = [out, in]
    if len(sa) == 2 and len(sb) == 2 and sa[0] == sb[1]:
        return (b.float() @ a.float()).to(dtype=b.dtype)

    # 情况 2: A=[in, r], B=[out, r] → B @ A^T = [out, r] @ [r, in] = [out, in]
    if len(sa) == 2 and len(sb) == 2 and sa[1] == sb[1]:
        return (b.float() @ a.float().T).to(dtype=b.dtype)

    # 情况 3: A=[r, in], B=[r, out] → B^T @ A = [out, r] @ [r, in] = [out, in]
    if len(sa) == 2 and len(sb) == 2 and sa[0] == sb[0]:
        return (b.float().T @ a.float()).to(dtype=b.dtype)

    # 情况 4: A=[in, r], B=[r, out] → (A@B)^T = [out, in]
    if len(sa) == 2 and len(sb) == 2 and sa[1] == sb[0]:
        return (a.float() @ b.float()).T.to(dtype=a.dtype)

    # Fallback: 暴力尝试
    print(f"  ⚠ 无法自动推断 matmul 方向: {a_key}{tuple(sa)} × {b_key}{tuple(sb)}，使用 B@A")
    return (b.float() @ a.float()).to(dtype=b.dtype)


def infer_rank(a: torch.Tensor, b: torch.Tensor) -> Optional[int]:
    """推断 LoRA rank (A 和 B 的共享维度)"""
    shared = set(a.shape) & set(b.shape)
    return min(shared) if shared else None


# ═══════════════════════════════════════════════════════════════
# 合并主逻辑
# ═══════════════════════════════════════════════════════════════

def merge_lora(
    weights: Dict[str, torch.Tensor],
    alpha: float = 1.0,
    base_weights: Optional[Dict[str, torch.Tensor]] = None,
) -> Dict[str, torch.Tensor]:
    """
    合并 LoRA 权重。

    流程:
      1. 分离 lora_A / lora_B / 非 lora key
      2. 配对同 base_name 的 A 和 B
      3. 计算 delta = lora_B @ lora_A，乘以 alpha/rank
      4. 如有 base model，将 delta 加到对应 base weight 上
      5. 非 lora key 原样保留
    """
    lora_a: Dict[str, torch.Tensor] = {}
    lora_b: Dict[str, torch.Tensor] = {}
    other: Dict[str, torch.Tensor] = {}

    for k, v in weights.items():
        if is_lora_a(k):
            lora_a[k] = v
        elif is_lora_b(k):
            lora_b[k] = v
        else:
            other[k] = v

    print(f"  LoRA-A: {len(lora_a)}  |  LoRA-B: {len(lora_b)}  |  其他: {len(other)}")

    if not lora_a and not lora_b:
        print("  ⚠ 未检测到 LoRA key，直接透传。")
        return dict(weights)

    # 按 base_name 分组
    a_by_base: Dict[str, Tuple[str, torch.Tensor]] = {}
    for k, v in lora_a.items():
        a_by_base[extract_base_name(k)] = (k, v)

    b_by_base: Dict[str, Tuple[str, torch.Tensor]] = {}
    for k, v in lora_b.items():
        b_by_base[extract_base_name(k)] = (k, v)

    merged: Dict[str, torch.Tensor] = dict(other)

    all_bases = sorted(set(a_by_base) | set(b_by_base))

    for base in all_bases:
        if base not in a_by_base:
            print(f"  ⚠ 孤立的 LoRA-B: {b_by_base[base][0]}")
            continue
        if base not in b_by_base:
            print(f"  ⚠ 孤立的 LoRA-A: {a_by_base[base][0]}")
            continue

        a_key, a_tensor = a_by_base[base]
        b_key, b_tensor = b_by_base[base]

        delta = compute_delta(a_tensor, b_tensor, a_key, b_key)

        # alpha / rank 缩放
        if alpha != 1.0:
            rank = infer_rank(a_tensor, b_tensor)
            scale = alpha / rank if rank else alpha
            delta = delta * scale

        # 输出 key: 去 lora 前缀, 加 .weight
        out_key = derive_output_key(base, a_key)
        merged[out_key] = delta

        print(f"  ✓ {base}  →  {out_key}  {tuple(delta.shape)}")

    # ── 合并到 base model ──
    if base_weights:
        merged = apply_to_base(merged, base_weights)

    return merged


def derive_output_key(base: str, a_key: str) -> str:
    """
    输出 key 命名。
  
    "model.layers.0.self_attn.q_proj" → "model.layers.0.self_attn.q_proj.weight"
    """
    if base.endswith((".weight", ".bias")):
        return base
    return f"{base}.weight"


def apply_to_base(
    deltas: Dict[str, torch.Tensor],
    base: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """将 delta 加到 base model 对应权重上。"""
    merged: Dict[str, torch.Tensor] = {}
    unmatched = 0

    for key, delta in deltas.items():
        # 直接匹配
        if key in base:
            merged[key] = base[key].to(delta.dtype) + delta
            continue
        # 尝试常见前缀变体
        for strip_prefix in ("base_model.model.", "model.", "backbone."):
            candidate = key.replace(strip_prefix, "")
            if candidate in base:
                merged[key] = base[candidate].to(delta.dtype) + delta
                break
        else:
            merged[key] = delta  # 找不到 base，保留 delta
            unmatched += 1
            print(f"  ⚠ 未找到 base weight: {key} (保留为 delta)")

    # 把 base 中有但 deltas 中没有的也带上
    delta_keys = set(deltas.keys())
    for bk, bv in base.items():
        if bk not in delta_keys:
            merged[bk] = bv

    if unmatched:
        print(f"  ⚠ {unmatched} 个 delta key 未找到对应 base weight")
    return merged


# ═══════════════════════════════════════════════════════════════
# 保存
# ═══════════════════════════════════════════════════════════════

def save_merged(weights: Dict[str, torch.Tensor], output: str, metadata: Optional[dict] = None):
    """通过 safetensors 导出。"""
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    save_safetensors(weights, output, metadata=metadata)


# ═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════

DTYPE_MAP = {
    "fp16": torch.float16,   "float16": torch.float16,
    "fp32": torch.float32,   "float32": torch.float32,
    "bf16": torch.bfloat16,  "bfloat16": torch.bfloat16,
}


def main():
    parser = argparse.ArgumentParser(
        description="LoRA 合并脚本 — 扫描 lora_A/lora_B 并合并导出 safetensors",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python lora_merge.py -i lora.safetensors -o merged.safetensors
  python lora_merge.py -i lora.pth -o merged.safetensors --alpha 16
  python lora_merge.py -i lora.pth -o merged.safetensors --base ./llama-7b --dtype bf16
  python lora_merge.py -i lora.safetensors -o merged.safetensors --base meta-llama/Llama-2-7b-hf
        """,
    )
    parser.add_argument("-i", "--input",  required=True,
                        help="输入权重文件 (.pth/.safetensors)")
    parser.add_argument("-o", "--output", required=True,
                        help="输出 safetensors 路径")
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="LoRA alpha 缩放因子, delta *= alpha/rank (默认: 1.0)")
    parser.add_argument("--base", type=str, default=None,
                        help="(可选) Base model 路径或 HF repo id，将 delta 合并进去")
    parser.add_argument("--dtype", type=str, default=None, choices=list(DTYPE_MAP),
                        help="输出数据类型 (fp16/fp32/bf16)，默认保持原样")
    parser.add_argument("--verbose", "-v", action="store_true")

    args = parser.parse_args()

    # ── 1. 加载 LoRA ──
    print(f"📂 加载 LoRA: {args.input}")
    weights = load_weights(args.input)
    print(f"   共 {len(weights)} 个 key")

    # ── 2. (可选) 加载 base model ──
    base_weights = None
    if args.base:
        print(f"📂 加载 Base Model: {args.base}")
        base_weights = load_base_model(args.base, verbose=args.verbose)
        print(f"   共 {len(base_weights)} 个 key")

    # ── 3. 合并 ──
    print(f"🔀 合并 LoRA (alpha={args.alpha}) ...")
    merged = merge_lora(weights, alpha=args.alpha, base_weights=base_weights)

    # ── 4. 类型转换 ──
    if args.dtype:
        target = DTYPE_MAP[args.dtype]
        print(f"🔄 转换为 {args.dtype} ...")
        merged = {k: v.to(dtype=target) for k, v in merged.items()}

    # ── 5. 导出 ──
    metadata = {
        "format": "pt",
        "merged_from_lora": "true",
    }
    if args.alpha != 1.0:
        metadata["lora_alpha"] = str(args.alpha)

    print(f"💾 导出: {args.output}")
    save_merged(merged, args.output, metadata=metadata)

    total = sum(v.numel() for v in merged.values())
    print(f"\n✅ 完成! {len(merged)} keys, {total:,} params → {args.output}")


if __name__ == "__main__":
    main()
```

---

## 脚本说明

### 核心流程

```
输入 (.pth / .safetensors)
        │
        ▼
  扫描 key → 分离 lora_A / lora_B / 其他
        │
        ▼
  按 base_name 配对 A ↔ B
        │
        ▼
  delta = lora_B @ lora_A    (自动推断 matmul 方向)
  delta *= alpha / rank      (可选缩放)
        │
        ▼
  (可选) + base model weights
        │
        ▼
  safetensors 导出
```

### 自动支持的 LoRA 格式

| 来源 | A key 示例 | B key 示例 |
|------|-----------|-----------|
| **PEFT / HF** | `q_proj.lora_A.weight` | `q_proj.lora_B.weight` |
| **Kohya** | `lora_te.fc1.lora_A.weight` | `lora_te.fc1.lora_B.weight` |
| **diffusers** | `to_q.lora_A.weight` | `to_q.lora_B.weight` |
| **自定义** | `xxx_lora_A_weight` | `xxx_lora_B_weight` |

### Matmul 自动推断

脚本会根据 shape 自动选择正确的乘法方式：

| A shape | B shape | 计算方式 |
|---------|---------|---------|
| `[r, in]` | `[out, r]` | `B @ A` → `[out, in]` |
| `[in, r]` | `[out, r]` | `B @ Aᵀ` → `[out, in]` |
| `[r, in]` | `[r, out]` | `Bᵀ @ A` → `[out, in]` |
| `[in, r]` | `[r, out]` | `(A @ B)ᵀ` → `[out, in]` |

### 使用示例

```bash
# 仅合并 delta（不合并 base model）
python lora_merge.py -i lora_weights.safetensors -o lora_delta.safetensors --alpha 16

# 合并到本地 base model
python lora_merge.py -i lora.pth -o full_model.safetensors --base ./Meta-Llama-3-8B --dtype bf16

# 合并到 HF Hub 模型
python lora_merge.py -i lora.safetensors -o full_model.safetensors \
    --base meta-llama/Llama-2-7b-hf --alpha 32 --dtype fp16
```

### 依赖

```bash
pip install torch safetensors  # 必需
pip install transformers       # 仅 --base 使用 HF repo 时需要
```