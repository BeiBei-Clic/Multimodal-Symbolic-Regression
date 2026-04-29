"""
潜空间嵌入向量范数分布分析

分析 SNIP 编码器学到的 512 维潜空间表示 encoded_y 的 L2 范数集中程度。
核心指标：变异系数 CV = std[||z||] / mean[||z||]，CV 越小表示范数越集中（球壳效应越强）。
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch

from experiments.pmlb.pmlb_inference import (
    build_inference_parser,
    configure_params,
    create_inference_components,
)


def build_latent_analysis_parser():
    parser = build_inference_parser()
    parser.set_defaults(
        beam_size=2,
        max_input_points=200,
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=5000,
        help="Number of synthetic samples to generate for analysis.",
    )
    # 编码专用 batch_size（configure_params 会把 params.batch_size 强制设为 1）
    parser.add_argument(
        "--encode_batch_size",
        type=int,
        default=32,
        help="Batch size for encoding (independent of params.batch_size).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./experiments/latent_analysis/output",
        help="Directory to save output PDF and stats.",
    )
    return parser


def collect_encoded_norms(env, model, n_samples, batch_size, device):
    """生成合成数据并收集 encoded_y 的范数统计量。"""
    all_norms = []
    all_sq_norms = []
    all_encoded_y = []
    n_collected = 0
    total_collected_full = 0
    max_full = 2000  # 只保留前 2000 个完整向量用于各维度统计

    while n_collected < n_samples:
        # 生成一个 batch 的有效样本
        batch_samples = []
        while len(batch_samples) < batch_size:
            expr, _ = env.gen_expr(train=True)
            if expr["tree"] is not None:
                batch_samples.append(expr)

        # 构造 embedder 输入：List[Sequence]，每个 Sequence 是 List[(x_ndarray, y_ndarray)]
        x1 = []
        for s in batch_samples:
            x_data = s["X_to_fit"][0]  # (n_points, input_dim)
            y_data = s["Y_to_fit"][0]  # (n_points, output_dim)
            seq = [(x_data[i], y_data[i]) for i in range(len(x_data))]
            x1.append(seq)

        with torch.no_grad():
            embeddings, lengths = model.embedder(x1)
            encoded_y = model.encoder_y(
                "fwd", x=embeddings, lengths=lengths, causal=False
            )
            # encoded_y shape: (batch_size, 512)

            norms = torch.norm(encoded_y, dim=1).cpu().numpy()
            sq_norms = (encoded_y**2).sum(dim=1).cpu().numpy()

            all_norms.append(norms)
            all_sq_norms.append(sq_norms)

            if total_collected_full < max_full:
                all_encoded_y.append(encoded_y.cpu().numpy())
                total_collected_full += len(batch_samples)

        n_collected += len(batch_samples)
        if n_collected % 500 < batch_size:
            print(f"  Collected {n_collected}/{n_samples} samples...")

    norms = np.concatenate(all_norms)[:n_samples]
    sq_norms = np.concatenate(all_sq_norms)[:n_samples]
    encoded_ys = np.concatenate(all_encoded_y, axis=0)[:max_full]

    return norms, sq_norms, encoded_ys


def print_statistics(norms, sq_norms, encoded_ys, d):
    """打印经验范数分布统计量，聚焦 CV 和球壳集中度。"""
    mean_norm = np.mean(norms)
    std_norm = np.std(norms)
    cv = std_norm / mean_norm
    p10, p25, p50, p75, p90 = np.percentile(norms, [10, 25, 50, 75, 90])

    print("=" * 60)
    print(f"潜空间维度 d = {d}")
    print(f"样本数 N = {len(norms)}")
    print("=" * 60)

    print(f"\n||z|| 分布统计:")
    print(f"  mean = {mean_norm:.4f}")
    print(f"  std  = {std_norm:.4f}")
    print(f"  CV (std/mean) = {cv:.4f}     ← 核心指标：越小越集中")
    print(f"  min = {norms.min():.4f}, max = {norms.max():.4f}")
    print(f"  P10 = {p10:.4f}, P90 = {p90:.4f}  ← 80% 区间宽度 = {p90 - p10:.4f}")
    print(f"  P25 = {p25:.4f}, P75 = {p75:.4f}  ← IQR = {p75 - p25:.4f}")

    within_1sigma = np.mean(
        (norms > mean_norm - std_norm) & (norms < mean_norm + std_norm)
    )
    within_10pct = np.mean(
        (norms > 0.9 * mean_norm) & (norms < 1.1 * mean_norm)
    )
    normed = norms / mean_norm
    print(f"\n球壳集中度:")
    print(f"  P(mean - std < ||z|| < mean + std) = {within_1sigma:.4f}")
    print(f"  P(0.9*mean < ||z|| < 1.1*mean) = {within_10pct:.4f}")
    print(f"  归一化范数 ||z||/mean 的 std = {normed.std():.4f}")

    print(f"\n参考：N(0,I_{d}) 的理论 CV ≈ 1/sqrt(2*{d}) ≈ {1.0 / np.sqrt(2 * d):.4f}")

    dim_means = encoded_ys.mean(axis=0)
    dim_vars = encoded_ys.var(axis=0)
    print(f"\n各维度 marginal 统计 (基于 {len(encoded_ys)} 个样本):")
    print(f"  维度均值分布: mean={dim_means.mean():.6f}, std={dim_means.std():.6f}")
    print(f"  维度方差分布: mean={dim_vars.mean():.6f}, std={dim_vars.std():.6f}")

    print("=" * 60)


def generate_plots(norms, d, output_dir):
    """画一张 ||z|| 直方图。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    pdf_path = output_path / "latent_norm_analysis.pdf"

    mean_norm = np.mean(norms)
    std_norm = np.std(norms)
    cv = std_norm / mean_norm

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(norms, bins=80, density=True, alpha=0.6, color="steelblue")
    ax.axvline(mean_norm, color="red", linestyle="--", lw=2,
               label=f"mean = {mean_norm:.2f}")
    ax.axvspan(mean_norm - std_norm, mean_norm + std_norm,
               alpha=0.2, color="orange",
               label=f"mean ± std ({std_norm:.2f})")
    ax.set_xlabel(r"$\|z\|$", fontsize=13)
    ax.set_ylabel("Density", fontsize=13)
    ax.set_title(
        rf"$\|z\|$ Distribution (d={d}, N={len(norms)}, CV={cv:.4f})",
        fontsize=14,
    )
    ax.legend(fontsize=11)
    fig.tight_layout()
    fig.savefig(pdf_path)
    plt.close(fig)

    print(f"\n图表已保存到: {pdf_path}")


def main():
    parser = build_latent_analysis_parser()
    params = configure_params(parser.parse_args())

    n_samples = params.n_samples
    encode_batch_size = params.encode_batch_size
    output_dir = params.output_dir
    device = params.device

    print(f"加载模型 (device={device})...")
    env, model = create_inference_components(params)
    env.rng = np.random.RandomState(42)

    d = 512
    print(f"收集 {n_samples} 个样本的潜空间编码 (encode_batch_size={encode_batch_size})...")
    norms, sq_norms, encoded_ys = collect_encoded_norms(
        env, model, n_samples, encode_batch_size, device
    )

    print(f"\n实际收集: norms={len(norms)}, sq_norms={len(sq_norms)}, encoded_ys={len(encoded_ys)}")
    print(f"encoded_y shape: {encoded_ys.shape}")

    print_statistics(norms, sq_norms, encoded_ys, d)
    generate_plots(norms, d, output_dir)


if __name__ == "__main__":
    main()
