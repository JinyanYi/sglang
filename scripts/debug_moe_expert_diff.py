"""
Compare MoE expert intermediate activations between GPU and NPU.

Workflow:
1. Run GPU server  → generates gpu_expert_dump_L001.pt
2. Run NPU server  → generates npu_expert_gmm1_L001.pt,
                              npu_expert_act_L001.pt,
                              npu_expert_gmm2_L001.pt,
                              npu_expert_routing_L001.pt,
                              npu_expert_finalize_L001.pt
3. Copy all .pt files to the same directory and run this script:
       python scripts/debug_moe_expert_diff.py [--dir /path/to/pt/files]

Expected output pinpoints which step first diverges:
  gmm1  -> npu_grouped_matmul gate_up  vs  F.linear gate_up
  act   -> npu_swiglu           vs  F.silu * up
  gmm2  -> npu_grouped_matmul down     vs  F.linear down
"""

import argparse
import os
import sys

import torch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def stat(t: torch.Tensor) -> str:
    t = t.float()
    return (f"norm={t.norm():.5f}  mean={t.mean():.6f}  "
            f"std={t.std():.6f}  sum={t.sum():.6f}  shape={list(t.shape)}")


def load(path: str) -> dict:
    if not os.path.exists(path):
        print(f"  [MISSING] {path}")
        return {}
    d = torch.load(path, map_location="cpu")
    print(f"  [OK]      {path}  keys={list(d.keys())}")
    return d


def compare(name: str, a: torch.Tensor, b: torch.Tensor) -> None:
    """Print norm-level comparison and element-wise diff (if shapes match)."""
    a, b = a.float(), b.float()
    print(f"\n{'='*60}")
    print(f"  Stage: {name}")
    print(f"  NPU   {stat(a)}")
    print(f"  GPU   {stat(b)}")
    ratio = a.norm() / (b.norm() + 1e-12)
    print(f"  norm ratio (NPU/GPU) = {ratio:.5f}")

    if a.shape == b.shape:
        diff = (a - b).abs()
        rel  = diff / (b.abs() + 1e-8)
        print(f"  max_abs_diff  = {diff.max():.6f}")
        print(f"  mean_abs_diff = {diff.mean():.6f}")
        print(f"  max_rel_diff  = {rel.max():.6f}")
        print(f"  mean_rel_diff = {rel.mean():.6f}")
        # Show worst offenders
        flat_diff = diff.flatten()
        topk_val, topk_idx = flat_diff.topk(min(5, flat_diff.numel()))
        print(f"  top-5 worst positions (flat index, npu_val, gpu_val, abs_diff):")
        for v, idx in zip(topk_val.tolist(), topk_idx.tolist()):
            print(f"    idx={idx:8d}  npu={a.flatten()[idx]:.6f}  "
                  f"gpu={b.flatten()[idx]:.6f}  diff={v:.6f}")
    else:
        print(f"  [SHAPE MISMATCH] NPU {list(a.shape)} vs GPU {list(b.shape)}")
        print("  Token ordering may differ between NPU and GPU dispatchers.")
        print("  Falling back to norm-only comparison.")


def compare_routing(npu_r: dict, gpu_r: dict) -> None:
    """Compare dispatcher outputs: dispatched_x, topk_ids, topk_weights."""
    print(f"\n{'='*60}")
    print("  Stage: routing (dispatcher output)")
    for key in ["dispatched_x", "topk_ids", "topk_weights"]:
        a = npu_r.get(key)
        b = gpu_r.get(key)
        if a is None or b is None:
            print(f"  [{key}] missing on one side, skipping")
            continue
        a, b = a.float(), b.float()
        if a.shape == b.shape:
            diff = (a - b).abs()
            print(f"  [{key}] NPU norm={a.norm():.5f}  GPU norm={b.norm():.5f}  "
                  f"max_diff={diff.max():.6f}  mean_diff={diff.mean():.6f}")
        else:
            print(f"  [{key}] shape mismatch: NPU={list(a.shape)} GPU={list(b.shape)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Compare GPU vs NPU MoE expert dumps")
    parser.add_argument("--dir", default=".", help="Directory containing the .pt files")
    parser.add_argument("--layer", default="001", help="Layer id string, e.g. 001")
    args = parser.parse_args()

    d   = args.dir
    lid = args.layer

    print(f"\nLoading .pt files from '{d}' for layer L{lid} ...\n")

    npu_routing  = load(os.path.join(d, f"npu_expert_routing_L{lid}.pt"))
    npu_gmm1_d   = load(os.path.join(d, f"npu_expert_gmm1_L{lid}.pt"))
    npu_act_d    = load(os.path.join(d, f"npu_expert_act_L{lid}.pt"))
    npu_gmm2_d   = load(os.path.join(d, f"npu_expert_gmm2_L{lid}.pt"))
    npu_final_d  = load(os.path.join(d, f"npu_expert_finalize_L{lid}.pt"))
    gpu_dump     = load(os.path.join(d, f"gpu_expert_dump_L{lid}.pt"))

    if not gpu_dump:
        print("\nERROR: gpu_expert_dump not found. Did the GPU server run with _lid==1 debug?")
        sys.exit(1)

    # --- routing ---
    if npu_routing and gpu_dump:
        npu_r = {k: npu_routing[k] for k in ["dispatched_x", "topk_ids", "topk_weights"]
                 if k in npu_routing}
        gpu_r = {k: gpu_dump[k] for k in ["dispatched_x", "topk_ids", "topk_weights"]
                 if k in gpu_dump}
        compare_routing(npu_r, gpu_r)

    # --- gmm1 (gate_up) ---
    if npu_gmm1_d and "gmm1_out" in gpu_dump:
        compare("gmm1 / gate_up  [npu_grouped_matmul vs F.linear]",
                npu_gmm1_d["gmm1_out"], gpu_dump["gmm1_out"])
    else:
        print("\n[SKIP] gmm1 comparison (missing data)")

    # --- act (swiglu) ---
    if npu_act_d and "act_out" in gpu_dump:
        compare("act / swiglu  [npu_swiglu vs F.silu(gate)*up]",
                npu_act_d["act_out"], gpu_dump["act_out"])
    else:
        print("\n[SKIP] act comparison (missing data)")

    # --- gmm2 (down) ---
    if npu_gmm2_d and "gmm2_out" in gpu_dump:
        compare("gmm2 / down  [npu_grouped_matmul vs F.linear]",
                npu_gmm2_d["gmm2_out"], gpu_dump["gmm2_out"])
    else:
        print("\n[SKIP] gmm2 comparison (missing data)")

    # --- finalize ---
    if npu_final_d:
        print(f"\n{'='*60}")
        print("  Stage: finalize (after npu_moe_finalize_routing)")
        print(f"  NPU   {stat(npu_final_d['finalize_out'])}")
        if "dispatched_x" in gpu_dump:
            print("  (GPU finalize not dumped separately; compare via [DBG EXPERT] log lines)")

    # --- summary ---
    print(f"\n{'='*60}")
    print("  DIAGNOSIS GUIDE")
    print("  ---------------")
    print("  If norm ratio deviates at stage:")
    print("    routing  -> dispatcher token order or topk_weights differ -> check npu_moe_init_routing_v2 args")
    print("    gmm1     -> npu_grouped_matmul gate_up is wrong           -> check w13 weight layout / dtype")
    print("    act      -> npu_swiglu gives different result             -> check swiglu split convention (gate first vs up first)")
    print("    gmm2     -> npu_grouped_matmul down is wrong              -> check w2 weight layout / dtype")
    print("    finalize -> npu_moe_finalize_routing weighting differs    -> check topk_weights scaling / drop_pad_mode")
    print()


if __name__ == "__main__":
    main()
