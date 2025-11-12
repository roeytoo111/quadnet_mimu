"""
Format evaluation results into publication-style tables (Markdown and LaTeX).

Usage:
    python3 src/format_tables.py --results_dir results --out_dir results/tables

It reads files matching *_metrics.json and *_eval.csv and writes tables summarizing
RMSE, MAE, max_error, std_error by experiment (split_mode_nIms).
"""
import json
import argparse
from pathlib import Path
import pandas as pd
import numpy as np


def load_metrics(results_dir: Path):
    rows = []

    # Load JSON metric files
    for path in sorted(results_dir.glob("*_metrics.json")):
        try:
            with open(path, 'r') as f:
                data = json.load(f)
        except Exception:
            continue

        # Infer experiment name from filename
        name = path.stem.replace('_metrics', '')

        if isinstance(data, dict) and 'distance' in data and 'altitude' in data:
            # target == both
            rows.append({'exp': name, 'target': 'distance', **data['distance']})
            rows.append({'exp': name, 'target': 'altitude', **data['altitude']})
        else:
            # single-target
            rows.append({'exp': name, 'target': 'distance', **data})

    # Also try to load *_eval.csv for per-trajectory breakdowns
    eval_rows = []
    for path in sorted(results_dir.glob("*_eval.csv")):
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        name = path.stem.replace('_eval', '')
        for _, r in df.iterrows():
            row = r.to_dict()
            row['exp'] = name
            eval_rows.append(row)

    return pd.DataFrame(rows), pd.DataFrame(eval_rows)


def format_markdown_table(df: pd.DataFrame, out_path: Path):
    if df.empty:
        print("No metrics found to format.")
        return

    lines = []
    header = ["Experiment", "Target", "RMSE", "MAE", "Max Error", "Std Error"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "---|" * len(header))

    for _, row in df.iterrows():
        exp = row['exp']
        target = row.get('target', 'distance')
        rmse = row.get('rmse', np.nan)
        mae = row.get('mae', np.nan)
        max_e = row.get('max_error', np.nan)
        std_e = row.get('std_error', np.nan)
        lines.append(f"| {exp} | {target} | {rmse:.4f} | {mae:.4f} | {max_e:.4f} | {std_e:.4f} |")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))
    print(f"Markdown table written to {out_path}")


def format_latex_table(df: pd.DataFrame, out_path: Path):
    """Write a simple LaTeX table summarizing metrics.

    We build the LaTeX newline programmatically to avoid complex escaping in the source.
    """
    if df.empty:
        print("No metrics found to format (LaTeX).")
        return

    bs = chr(92)
    nl = bs + bs

    lines = []
    lines.append("\\begin{tabular}{llrrrr}")
    lines.append("\\toprule")
    # Header row - end with LaTeX linebreak (\\)
    lines.append("Experiment & Target & RMSE & MAE & Max Error & Std Error " + r"\\")
    lines.append("\\midrule")

    for _, row in df.iterrows():
        exp = row['exp']
        target = row.get('target', 'distance')
        rmse = row.get('rmse', np.nan)
        mae = row.get('mae', np.nan)
        max_e = row.get('max_error', np.nan)
        std_e = row.get('std_error', np.nan)
        # Each printed row should end with \\ in source to produce \\ in output
        lines.append(f"{exp} & {target} & {rmse:.4f} & {mae:.4f} & {max_e:.4f} & {std_e:.4f} " + r"\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))
    print(f"LaTeX table written to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str, default='results')
    parser.add_argument('--out_dir', type=str, default='results/tables')
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir)

    metrics_df, eval_df = load_metrics(results_dir)

    md_path = out_dir / 'metrics_table.md'
    tex_path = out_dir / 'metrics_table.tex'

    format_markdown_table(metrics_df, md_path)
    format_latex_table(metrics_df, tex_path)

    # Also write per-trajectory CSV summary if available
    if not eval_df.empty:
        per_traj = eval_df.groupby(['exp', 'trajectory_id']).mean().reset_index()
        per_traj.to_csv(out_dir / 'per_trajectory_metrics.csv', index=False)
        print(f"Per-trajectory CSV written to {out_dir / 'per_trajectory_metrics.csv'}")


if __name__ == '__main__':
    main()
"""
Format evaluation results into publication-style tables (Markdown and LaTeX).

Usage:
    python3 src/format_tables.py --results_dir results --out_dir results/tables

It reads files matching *_metrics.json and *_eval.csv and prints/writes tables summarizing
RMSE, MAE, max_error, std_error by experiment (split_mode_nIms).
"""
import json
import argparse
from pathlib import Path
import pandas as pd
import numpy as np


def load_metrics(results_dir: Path):
    rows = []

    # Load JSON metric files
    for path in sorted(results_dir.glob("*_metrics.json")):
        try:
            with open(path, 'r') as f:
                data = json.load(f)
        except Exception:
            continue

        # Infer experiment name from filename
        name = path.stem.replace('_metrics', '')

        if isinstance(data, dict) and 'distance' in data and 'altitude' in data:
            # target == both
            rows.append({'exp': name, 'target': 'distance', **data['distance']})
            rows.append({'exp': name, 'target': 'altitude', **data['altitude']})
        else:
            # single-target
            rows.append({'exp': name, 'target': 'distance', **data})

    # Also try to load *_eval.csv for per-trajectory breakdowns
    eval_rows = []
    for path in sorted(results_dir.glob("*_eval.csv")):
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        name = path.stem.replace('_eval', '')
        for _, r in df.iterrows():
            row = r.to_dict()
            row['exp'] = name
            eval_rows.append(row)

    return pd.DataFrame(rows), pd.DataFrame(eval_rows)


def format_markdown_table(df: pd.DataFrame, out_path: Path):
    # Pivot experiments x metrics
    if df.empty:
        print("No metrics found to format.")
        return

    # For each experiment and target, create a formatted row
    lines = []
    header = ["Experiment", "Target", "RMSE", "MAE", "Max Error", "Std Error"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "---|" * len(header))

    for _, row in df.iterrows():
        exp = row['exp']
        target = row.get('target', 'distance')
        rmse = row.get('rmse', np.nan)
        mae = row.get('mae', np.nan)
        max_e = row.get('max_error', np.nan)
        std_e = row.get('std_error', np.nan)
        lines.append(f"| {exp} | {target} | {rmse:.4f} | {mae:.4f} | {max_e:.4f} | {std_e:.4f} |")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))
    print(f"Markdown table written to {out_path}")


def format_latex_table(df: pd.DataFrame, out_path: Path):
    """
    Format evaluation results into publication-style tables (Markdown and LaTeX).

    Usage:
        python3 src/format_tables.py --results_dir results --out_dir results/tables

    It reads files matching *_metrics.json and *_eval.csv and writes tables summarizing
    RMSE, MAE, max_error, std_error by experiment (split_mode_nIms).
    """
    import json
    import argparse
    from pathlib import Path
    import pandas as pd
    import numpy as np


    def load_metrics(results_dir: Path):
        rows = []

        # Load JSON metric files
        for path in sorted(results_dir.glob("*_metrics.json")):
            try:
                with open(path, 'r') as f:
                    data = json.load(f)
            except Exception:
                continue

            # Infer experiment name from filename
            name = path.stem.replace('_metrics', '')

            if isinstance(data, dict) and 'distance' in data and 'altitude' in data:
                # target == both
                rows.append({'exp': name, 'target': 'distance', **data['distance']})
                rows.append({'exp': name, 'target': 'altitude', **data['altitude']})
            else:
                # single-target
                rows.append({'exp': name, 'target': 'distance', **data})

        # Also try to load *_eval.csv for per-trajectory breakdowns
        eval_rows = []
        for path in sorted(results_dir.glob("*_eval.csv")):
            try:
                df = pd.read_csv(path)
            except Exception:
                continue
            name = path.stem.replace('_eval', '')
            for _, r in df.iterrows():
                row = r.to_dict()
                row['exp'] = name
                eval_rows.append(row)

        return pd.DataFrame(rows), pd.DataFrame(eval_rows)


    def format_markdown_table(df: pd.DataFrame, out_path: Path):
        # Pivot experiments x metrics
        if df.empty:
            print("No metrics found to format.")
            return

        # For each experiment and target, create a formatted row
        lines = []
        header = ["Experiment", "Target", "RMSE", "MAE", "Max Error", "Std Error"]
        lines.append("| " + " | ".join(header) + " |")
        lines.append("|" + "---|" * len(header))

        for _, row in df.iterrows():
            exp = row['exp']
            target = row.get('target', 'distance')
            rmse = row.get('rmse', np.nan)
            mae = row.get('mae', np.nan)
            max_e = row.get('max_error', np.nan)
            std_e = row.get('std_error', np.nan)
            lines.append(f"| {exp} | {target} | {rmse:.4f} | {mae:.4f} | {max_e:.4f} | {std_e:.4f} |")

        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("\n".join(lines))
        print(f"Markdown table written to {out_path}")


    def format_latex_table(df: pd.DataFrame, out_path: Path):
        """Write a simple LaTeX table summarizing metrics.

        Each row ends with a LaTeX newline `\\` so we escape that in Python as "\\\\".
        """
        if df.empty:
            print("No metrics found to format (LaTeX).")
            return

        lines = []
        lines.append("\\begin{tabular}{llrrrr}")
        lines.append("\\toprule")
        lines.append("Experiment & Target & RMSE & MAE & Max Error & Std Error " + r"\\")
        lines.append("\\midrule")

        for _, row in df.iterrows():
            exp = row['exp']
            target = row.get('target', 'distance')
            rmse = row.get('rmse', np.nan)
            mae = row.get('mae', np.nan)
            max_e = row.get('max_error', np.nan)
            std_e = row.get('std_error', np.nan)
            # Each printed row should end with \\ in source to produce \\ in output
            lines.append(f"{exp} & {target} & {rmse:.4f} & {mae:.4f} & {max_e:.4f} & {std_e:.4f} " + r"\\")

        lines.append("\\bottomrule")
        lines.append("\\end{tabular}")

        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("\n".join(lines))
        print(f"LaTeX table written to {out_path}")


    def main():
        parser = argparse.ArgumentParser()
        parser.add_argument('--results_dir', type=str, default='results')
        parser.add_argument('--out_dir', type=str, default='results/tables')
        args = parser.parse_args()

        results_dir = Path(args.results_dir)
        out_dir = Path(args.out_dir)

        metrics_df, eval_df = load_metrics(results_dir)

        md_path = out_dir / 'metrics_table.md'
        tex_path = out_dir / 'metrics_table.tex'

        format_markdown_table(metrics_df, md_path)
        format_latex_table(metrics_df, tex_path)

        # Also write per-trajectory CSV summary if available
        if not eval_df.empty:
            per_traj = eval_df.groupby(['exp', 'trajectory_id']).mean().reset_index()
            per_traj.to_csv(out_dir / 'per_trajectory_metrics.csv', index=False)
            print(f"Per-trajectory CSV written to {out_dir / 'per_trajectory_metrics.csv'}")


    if __name__ == '__main__':
        main()
