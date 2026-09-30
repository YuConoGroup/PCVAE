"""3D Pareto-front analysis and visualisation for the PCVAE property predictions.

Computes the 3-objective Pareto front (maximize AIP, minimize MIC and toxicity)
for a round's normalized prediction CSV, writes the optimal/dominated solution
tables, and renders a 3D front plot plus 2D projections.

Usage
-----
    python HV3D2.py --input input/normalized_with_seq_round4.csv --out-dir png --round-tag 4
"""
import argparse
import os

import matplotlib
matplotlib.use('Agg')  # headless rendering
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def calculate_pareto_front_3d(data, minimize_cols, maximize_cols):
    """Compute the 3D Pareto front (chunked NumPy vectorisation, handles tens of thousands of points).

    Parameters
    ----------
    data : DataFrame with the objective columns.
    minimize_cols : columns to minimize.
    maximize_cols : columns to maximize.

    Returns
    -------
    (pareto_optimal, dominated) : DataFrames of non-dominated and dominated solutions.
    """
    df = data.copy()

    # Convert maximization objectives to minimization by negation.
    comparison_cols = list(minimize_cols)
    neg_cols = []
    for col in maximize_cols:
        neg_col = f'{col}_min'
        df[neg_col] = -df[col]
        comparison_cols.append(neg_col)
        neg_cols.append(neg_col)

    objectives = df[comparison_cols].to_numpy(dtype=float)
    n = len(objectives)
    is_dominated = np.zeros(n, dtype=bool)

    # Chunked vectorisation: mark a point dominated if any point is <= in all
    # dimensions and < in at least one.
    chunk = 512
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        block = objectives[start:end]          # (m, 3)
        le_all = np.ones((end - start, n), dtype=bool)   # all dimensions <=
        lt_any = np.zeros((end - start, n), dtype=bool)  # at least one dimension <
        for d in range(objectives.shape[1]):
            obj_d = objectives[:, d]
            blk_d = block[:, d]
            le_all &= obj_d[None, :] <= blk_d[:, None]
            lt_any |= obj_d[None, :] < blk_d[:, None]
        is_dominated[start:end] = (le_all & lt_any).any(axis=1)

    pareto_optimal = df[~is_dominated].copy()
    dominated = df[is_dominated].copy()

    # Drop temporary columns.
    for col in neg_cols:
        pareto_optimal.drop(col, axis=1, inplace=True, errors='ignore')
        dominated.drop(col, axis=1, inplace=True, errors='ignore')

    return pareto_optimal, dominated


def plot_3d_pareto_front(pareto_optimal, dominated, minimize_cols, maximize_cols, save_path):
    """Render the 3D Pareto front."""
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # X axis: maximization objective; Y/Z axes: minimization objectives.
    x_data = pareto_optimal[maximize_cols[0]]
    y_data = pareto_optimal[minimize_cols[0]]
    z_data = pareto_optimal[minimize_cols[1]]

    ax.set_xlabel(f'{maximize_cols[0]} (maximize)', fontsize=12, labelpad=10)
    ax.set_ylabel(f'{minimize_cols[0]} (minimize)', fontsize=12, labelpad=10)
    ax.set_zlabel(f'{minimize_cols[1]} (minimize)', fontsize=12, labelpad=10)

    if len(dominated) > 0:
        ax.scatter(dominated[maximize_cols[0]],
                   dominated[minimize_cols[0]],
                   dominated[minimize_cols[1]],
                   c='gray', alpha=0.15, s=5, label=f'Dominated (n={len(dominated)})')

    if len(pareto_optimal) > 0:
        scatter = ax.scatter(x_data, y_data, z_data,
                             c=x_data, cmap='viridis',
                             s=80, alpha=0.9, edgecolors='black',
                             linewidths=0.5,
                             label=f'Pareto Optimal (n={len(pareto_optimal)})')

        cbar = fig.colorbar(scatter, ax=ax, pad=0.1)
        cbar.set_label(maximize_cols[0], fontsize=10)

    ax.set_title('3D Pareto Front', fontsize=14, fontweight='bold', pad=20)

    # Viewing angle for best readability.
    ax.view_init(elev=25, azim=45)

    ax.legend(loc='upper right', fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Pareto front figure saved to: {save_path}")


def plot_2d_projections(pareto_optimal, dominated, minimize_cols, maximize_cols, output_dir):
    """Render the three pairwise 2D projections plus a statistics panel."""
    max_c = maximize_cols[0]
    min1, min2 = minimize_cols[0], minimize_cols[1]

    # (x_col, y_col, x direction, y direction)
    views = [
        (max_c, min1, 'maximize', 'minimize'),
        (min1, min2, 'minimize', 'minimize'),
        (max_c, min2, 'maximize', 'minimize'),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Pareto Front 2D Projections', fontsize=16, fontweight='bold')

    for ax, (xc, yc, xd, yd) in zip(axes.flat[:3], views):
        if len(dominated) > 0:
            ax.scatter(dominated[xc], dominated[yc],
                       c='gray', alpha=0.15, s=5, label='Dominated')
        if len(pareto_optimal) > 0:
            ax.scatter(pareto_optimal[xc], pareto_optimal[yc],
                       c=pareto_optimal[max_c], cmap='viridis',
                       s=40, alpha=0.9, edgecolors='black', linewidths=0.3)
        ax.set_title(f'{yc} vs {xc}')
        ax.set_xlabel(f'{xc} ({xd})')
        ax.set_ylabel(f'{yc} ({yd})')
        ax.legend(loc='best')

    # Statistics panel.
    ax = axes[1, 1]
    ax.axis('off')
    if len(pareto_optimal) > 0:
        stats_text = f'Pareto optimal: {len(pareto_optimal)}\n'
        stats_text += f'Dominated: {len(dominated)}\n\n'
        for col in [max_c, min1, min2]:
            s = pareto_optimal[col]
            stats_text += f'{col} range:\n'
            stats_text += f'  min: {s.min():.4f}\n  max: {s.max():.4f}\n\n'

        ax.text(0.1, 0.5, stats_text, fontsize=10, fontfamily='monospace',
                bbox=dict(facecolor='lightyellow', alpha=0.8, boxstyle='round,pad=1'))

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'pareto_2d_projections.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("2D projection figure saved")


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description="3D Pareto front analysis for PCVAE predictions")
    ap.add_argument("--input", default=os.path.join(base_dir, "input", "normalized_with_seq_round4.csv"),
                    help="normalized prediction CSV (columns: AIP_norm, TOXIN_norm, MIC_norm)")
    ap.add_argument("--out-dir", default=os.path.join(base_dir, "png"),
                    help="output directory for figures and tables")
    ap.add_argument("--round-tag", default="4", help="round tag used in output filenames")
    args = ap.parse_args()

    input_file = args.input
    output_dir = args.out_dir
    round_tag = args.round_tag

    os.makedirs(output_dir, exist_ok=True)

    try:
        print("Reading CSV file...")
        df = pd.read_csv(input_file)
        print(f"Data loaded: {len(df)} records")

        # Columns produced by process.py normalization.
        required_columns = ['AIP_norm', 'TOXIN_norm', 'MIC_norm']
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            print(f"Error: CSV is missing columns: {missing_columns}")
            print(f"Columns present: {list(df.columns)}")
            return

        # Optimization directions (after normalization: higher AIP_norm is
        # better; lower TOXIN_norm and MIC_norm are better).
        minimize_cols = ['TOXIN_norm', 'MIC_norm']
        maximize_cols = ['AIP_norm']

        print("Computing Pareto front...")
        pareto_optimal, dominated = calculate_pareto_front_3d(df, minimize_cols, maximize_cols)

        print(f"Pareto-optimal solutions: {len(pareto_optimal)}")
        print(f"Dominated solutions: {len(dominated)}")

        pareto_file = os.path.join(output_dir, f'pareto_optimal_solutions_round{round_tag}.csv')
        dominated_file = os.path.join(output_dir, f'dominated_solutions_round{round_tag}.csv')
        summary_file = os.path.join(output_dir, f'pareto_summary_round{round_tag}.txt')

        pareto_optimal.to_csv(pareto_file, index=False)
        dominated.to_csv(dominated_file, index=False)

        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(f"===== Pareto front analysis summary (round{round_tag}) =====\n\n")
            f.write(f"Input file: {input_file}\n")
            f.write(f"Total samples: {len(df)}\n")
            f.write(f"Pareto-optimal solutions: {len(pareto_optimal)}\n")
            f.write(f"Dominated solutions: {len(dominated)}\n\n")

            f.write("Pareto-optimal solution statistics:\n")
            f.write("-" * 50 + "\n")
            for col in ['AIP_norm', 'TOXIN_norm', 'MIC_norm']:
                f.write(f"{col}:\n")
                f.write(f"  min: {pareto_optimal[col].min():.6f}\n")
                f.write(f"  max: {pareto_optimal[col].max():.6f}\n")
                f.write(f"  mean: {pareto_optimal[col].mean():.6f}\n")
                f.write(f"  std: {pareto_optimal[col].std():.6f}\n\n")

        print("Results saved to CSV files")

        plot_file = os.path.join(output_dir, f'pareto_front_3d_round{round_tag}.png')
        plot_3d_pareto_front(pareto_optimal, dominated, minimize_cols, maximize_cols, plot_file)

        plot_2d_projections(pareto_optimal, dominated, minimize_cols, maximize_cols, output_dir)

        print("All analyses complete.")

    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
