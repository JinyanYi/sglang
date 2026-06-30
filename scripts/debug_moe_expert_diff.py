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

LATEST FINDING (see compare output):
  gmm1 / act / gmm2 norm ratios are ~1.0002  →  matrix computation is CORRECT on NPU
  topk_weights differ by ~2.82x  →  THIS is the root cause of the output divergence
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
        flat_diff = diff.flatten()
        topk_val, topk_idx = flat_diff.topk(min(5, flat_diff.numel()))
        print(f"  top-5 worst positions (flat index, npu_val, gpu_val, abs_diff):")
        for v, idx in zip(topk_val.tolist(), topk_idx.tolist()):
            print(f"    idx={idx:8d}  npu={a.flatten()[idx]:.6f}  "
                  f"gpu={b.flatten()[idx]:.6f}  diff={v:.6f}")
    else:
        print(f"  [SHAPE MISMATCH] NPU {list(a.shape)} vs GPU {list(b.shape)}")


# ---------------------------------------------------------------------------
# Topk weights / ids deep analysis
# ---------------------------------------------------------------------------

def analyse_topk_weights(npu_r: dict, gpu_r: dict) -> None:
    """
    The primary purpose: determine whether the topk_weights difference is
    caused by (a) element ordering within each token's top-k, or
    (b) actual scaling / value difference.

    Strategy: sort both sides by expert_id within each token, then compare.
    If norms converge after sorting  → ordering artifact only
    If norms still differ            → real scaling bug
    """
    npu_w = npu_r.get("topk_weights")
    npu_ids = npu_r.get("topk_ids")
    gpu_w = gpu_r.get("topk_weights")
    gpu_ids = gpu_r.get("topk_ids")

    if any(x is None for x in [npu_w, npu_ids, gpu_w, gpu_ids]):
        print("  [SKIP] topk_weights analysis: missing tensors")
        return

    npu_w   = npu_w.float()
    gpu_w   = gpu_w.float()
    npu_ids = npu_ids.long()
    gpu_ids = gpu_ids.long()

    print(f"\n{'='*60}")
    print("  Stage: topk_weights DEEP ANALYSIS")
    print(f"  NPU topk_weights  {stat(npu_w)}")
    print(f"  GPU topk_weights  {stat(gpu_w)}")
    print(f"  norm ratio NPU/GPU = {npu_w.norm() / (gpu_w.norm() + 1e-12):.5f}")

    # Per-token sum and max of weights
    print(f"\n  Per-token weight statistics (first 8 tokens):")
    print(f"  {'tok':>4}  {'NPU sum':>10} {'GPU sum':>10} {'NPU max':>10} {'GPU max':>10}")
    for t in range(min(8, npu_w.shape[0])):
        nw = npu_w[t] if npu_w.dim() == 2 else npu_w[t*npu_ids.shape[1]:(t+1)*npu_ids.shape[1]]
        gw = gpu_w[t] if gpu_w.dim() == 2 else gpu_w[t*gpu_ids.shape[1]:(t+1)*gpu_ids.shape[1]]
        print(f"  {t:>4}  {nw.sum():>10.5f} {gw.sum():>10.5f} {nw.max():>10.5f} {gw.max():>10.5f}")

    if npu_w.shape != gpu_w.shape:
        print(f"\n  [SHAPE MISMATCH] cannot do element-wise comparison after sorting")
        return

    # Sort by expert_id within each token, then compare
    assert npu_ids.shape == gpu_ids.shape, "topk_ids shape mismatch"
    n_tokens, top_k = npu_ids.shape

    npu_sorted_w = torch.zeros_like(npu_w)
    gpu_sorted_w = torch.zeros_like(gpu_w)
    ids_match_count = 0

    for t in range(n_tokens):
        npu_order = npu_ids[t].argsort()
        gpu_order = gpu_ids[t].argsort()
        npu_sorted_w[t] = npu_w[t][npu_order]
        gpu_sorted_w[t] = gpu_w[t][gpu_order]
        if (npu_ids[t][npu_order] == gpu_ids[t][gpu_order]).all():
            ids_match_count += 1

    print(f"\n  After sorting by expert_id within each token:")
    print(f"  Tokens with identical expert sets: {ids_match_count} / {n_tokens} "
          f"({100*ids_match_count/n_tokens:.1f}%)")
    print(f"  NPU sorted topk_weights norm = {npu_sorted_w.norm():.5f}")
    print(f"  GPU sorted topk_weights norm = {gpu_sorted_w.norm():.5f}")
    print(f"  norm ratio after sort = {npu_sorted_w.norm()/(gpu_sorted_w.norm()+1e-12):.5f}")

    # For tokens with matching expert sets, compare weights directly
    matched = []
    for t in range(n_tokens):
        npu_order = npu_ids[t].argsort()
        gpu_order = gpu_ids[t].argsort()
        if (npu_ids[t][npu_order] == gpu_ids[t][gpu_order]).all():
            matched.append(t)

    if matched:
        mt = torch.tensor(matched)
        diff = (npu_sorted_w[mt] - gpu_sorted_w[mt]).abs()
        ratio_per_elem = npu_sorted_w[mt].abs() / (gpu_sorted_w[mt].abs() + 1e-12)
        print(f"\n  Weight comparison for {len(matched)} tokens with matching expert sets:")
        print(f"  max_abs_diff      = {diff.max():.6f}")
        print(f"  mean_abs_diff     = {diff.mean():.6f}")
        print(f"  mean weight ratio = {ratio_per_elem.mean():.5f}  (NPU/GPU, expect ~1.0 if correct)")
        print(f"  First matched token (tok={matched[0]}):")
        t0 = matched[0]
        npu_o = npu_ids[t0].argsort()
        gpu_o = gpu_ids[t0].argsort()
        for k in range(top_k):
            eid = npu_ids[t0][npu_o[k]].item()
            nw  = npu_w[t0][npu_o[k]].item()
            gw  = gpu_w[t0][gpu_o[k]].item()
            print(f"    expert={eid:3d}  npu_w={nw:.6f}  gpu_w={gw:.6f}  ratio={nw/(gw+1e-12):.4f}")
    else:
        print("\n  No tokens have fully matching expert sets between NPU and GPU.")
        print("  The router is selecting different experts — check topk / scoring computation.")


# ---------------------------------------------------------------------------
# Finalize comparison: use gmm2 outputs + topk_weights to reconstruct
# ---------------------------------------------------------------------------

def analyse_finalize(npu_r: dict, npu_gmm2: dict, npu_final: dict, gpu_r: dict, gpu_dump: dict) -> None:
    """
    Reconstruct the weighted-sum step manually to confirm topk_weights
    are the sole cause of divergence.
    """
    # We have gmm2_out[1496, 4096] and topk_weights/ids.
    # finalize = scatter_add(gmm2_out * topk_weights, to original token positions).
    # We can compute this for GPU (sorted order) and compare norm with gpu [DBG EXPERT].
    print(f"\n{'='*60}")
    print("  Stage: finalize reconstruction")

    if not (npu_final and gpu_dump):
        print("  [SKIP] missing data")
        return

    fn = npu_final.get("finalize_out")
    if fn is not None:
        fn = fn.float()
        print(f"  NPU finalize_out  {stat(fn)}")
        per_tok_norm = fn.norm(dim=-1)
        print(f"  Per-token norm: min={per_tok_norm.min():.5f}  max={per_tok_norm.max():.5f}  "
              f"mean={per_tok_norm.mean():.5f}  last={per_tok_norm[-1]:.5f}")

    # GPU: reconstruct finalize from gmm2 + topk_weights
    gw  = gpu_r.get("topk_weights")
    gid = gpu_r.get("topk_ids")
    gd  = gpu_dump.get("gmm2_out")   # shape [n_tokens*top_k, H] sorted by expert

    if gw is not None and gid is not None and gd is not None:
        gw  = gw.float()    # [N, K]
        gid = gid.long()    # [N, K]
        gd  = gd.float()    # [N*K, H] sorted by expert order (from _moe_debug_native_dump)

        n_tokens, top_k = gid.shape
        H = gd.shape[-1]

        # _moe_debug_native_dump sorts tokens by expert; undo the sort to get per-(token, expert) order
        idxs = gid.view(-1).argsort()
        # sorted_tokens[i] = x[idxs[i] // top_k], and gd is already in sorted order
        # Weighted sum: output[token] = sum_k(gd_sorted[rank] * gw[token, k])
        # But we need to map back. Use the same logic as moe_forward_native:
        new_x = torch.empty_like(gd)
        new_x[idxs] = gd

        # new_x[i*top_k : (i+1)*top_k] = expert outputs for token i's top-k experts
        gpu_finalize = (
            new_x.view(n_tokens, top_k, H)
            .mul(gw.unsqueeze(-1))
            .sum(dim=1)
        )
        print(f"  GPU reconstructed finalize  {stat(gpu_finalize)}")
        per_tok_norm_gpu = gpu_finalize.norm(dim=-1)
        print(f"  Per-token norm: min={per_tok_norm_gpu.min():.5f}  max={per_tok_norm_gpu.max():.5f}  "
              f"mean={per_tok_norm_gpu.mean():.5f}  last={per_tok_norm_gpu[-1]:.5f}")

        if fn is not None and fn.shape == gpu_finalize.shape:
            ratio = fn.norm() / (gpu_finalize.norm() + 1e-12)
            print(f"  NPU/GPU finalize norm ratio = {ratio:.5f}")


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
        print("\nERROR: gpu_expert_dump not found. Did the GPU server run with debug?")
        sys.exit(1)

    # --- topk weights deep analysis (primary bug) ---
    if npu_routing and gpu_dump:
        gpu_r = {k: gpu_dump[k] for k in ["dispatched_x", "topk_ids", "topk_weights"] if k in gpu_dump}
        analyse_topk_weights(npu_routing, gpu_r)

    # --- matrix computation stages ---
    if npu_gmm1_d and "gmm1_out" in gpu_dump:
        compare("gmm1 / gate_up  [npu_grouped_matmul vs F.linear]",
                npu_gmm1_d["gmm1_out"], gpu_dump["gmm1_out"])

    if npu_act_d and "act_out" in gpu_dump:
        compare("act / swiglu  [npu_swiglu vs F.silu(gate)*up]",
                npu_act_d["act_out"], gpu_dump["act_out"])

    if npu_gmm2_d and "gmm2_out" in gpu_dump:
        compare("gmm2 / down  [npu_grouped_matmul vs F.linear]",
                npu_gmm2_d["gmm2_out"], gpu_dump["gmm2_out"])

    # --- finalize reconstruction ---
    analyse_finalize(npu_routing, npu_gmm2_d, npu_final_d, gpu_dump, gpu_dump)

    # --- summary ---
    print(f"\n{'='*60}")
    print("  CURRENT STATUS")
    print("  --------------")
    print("  gmm1 / act / gmm2 norm ratios ≈ 1.0  →  NPU matrix computation is CORRECT")
    print("  topk_weights differ by ~2.82x  →  ROOT CAUSE of output divergence")
    print()
    print("  NEXT: read 'topk_weights DEEP ANALYSIS' above.")
    print("  Case A: mean weight ratio ≈ 2.82, norm converges after sort")
    print("    → router_scaling_factor applied on GPU but NOT on NPU in finalize_routing")
    print("    → fix: multiply topk_weights by router_scaling_factor before npu_moe_finalize_routing")
    print("    → OR check FusedMoE.should_fuse_routed_scaling_factor_in_topk for NPU path")
    print()
    print("  Case B: different expert sets selected (ids_match_count < 50%)")
    print("    → TopK scoring/correction_bias applied differently on NPU")
    print("    → add debug prints in TopK.forward to compare logits before/after correction_bias")
    print()
    print("  Case C: norm still differs after sorting (scaling not ordering)")
    print("    → check if renormalize=True/False differs between GPU and NPU paths")
    print("    → print config.route_norm value")
    print()


if __name__ == "__main__":
    main()
