import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def _bh_qvalues(pvals: np.ndarray) -> np.ndarray:
    """Return Benjamini-Hochberg FDR-adjusted q-values."""
    pvals = np.asarray(pvals, dtype=float)
    n = len(pvals)
    order = np.argsort(pvals)
    ranks = np.empty(n, int)
    ranks[order] = np.arange(1, n + 1)
    qvals = pvals * n / ranks
    qvals[order[::-1]] = np.minimum.accumulate(qvals[order[::-1]])
    return np.minimum(qvals, 1.0)


def _parse_condition(name: str) -> tuple[str, str, float]:
    """Return scenario, parameter label, and numeric value from a condition name."""
    m = re.search(r"^(linear|interaction|quadratic)_(beta|w)_([0-9.]+)", name)
    if m:
        return m.group(1), m.group(2), float(m.group(3))
    return name, "param", float("nan")


def compute_metrics(result_dir: str, true_groups=None, alpha: float = 0.05):
    """Compute empirical power and FDR from pvalue.csv files.

    Parameters
    ----------
    result_dir : str
        Root directory containing simulation results.
        Each simulation may have subdirectories with experiments
        that contain ``pvalue.csv``.
    true_groups : list[str], optional
        Pathway names with a true association. Defaults to
        ["map00400", "map00860"].
    alpha : float, default 0.05
        Significance threshold for declaring discoveries after
        Benjamini-Hochberg FDR correction.

    Saves per-experiment ``empirical_power_<cond>.png`` and ``fdr_<cond>.png``
    in ``result_dir`` and prints a summary table.
    """
    if true_groups is None:
        true_groups = ["map00400", "map00860"]

    pval_by_exp: dict[str, list[pd.DataFrame]] = {}
    for sim in sorted(os.listdir(result_dir)):
        sim_dir = os.path.join(result_dir, sim)
        if not os.path.isdir(sim_dir):
            continue
        for exp in sorted(os.listdir(sim_dir)):
            pv_path = os.path.join(sim_dir, exp, "pvalue.csv")
            if not os.path.exists(pv_path):
                continue
            df = pd.read_csv(pv_path)
            df["simulation"] = sim
            pval_by_exp.setdefault(exp, []).append(df)

    if not pval_by_exp:
        print("No pvalue.csv files were found in the given directory.")
        return

    summary_rows = []
    scenario_data: dict[str, dict] = {}

    for exp, tables in sorted(pval_by_exp.items()):
        df_all = pd.concat(tables, ignore_index=True)
        df_all["qvalue"] = _bh_qvalues(df_all["pvalue"].values)
        df_all["reject"] = df_all["qvalue"] <= alpha

        power_df = (
            df_all[df_all["pathway"].isin(true_groups)]
            .groupby("pathway")["reject"]
            .mean()
            .reset_index(name="empirical_power")
        )

        total_rejects = int(df_all["reject"].sum())
        false_rejects = df_all[~df_all["pathway"].isin(true_groups) & df_all["reject"]]
        fdr_value = len(false_rejects) / total_rejects if total_rejects else 0.0

        scen, label, val = _parse_condition(exp)
        info = scenario_data.setdefault(scen, {"label": label, "vals": []})
        info["vals"].append((val, power_df, fdr_value))

        # Plot empirical power for this experiment
        ax = power_df.plot(kind="bar", x="pathway", y="empirical_power", legend=False)
        ax.set_xlabel("Pathway")
        ax.set_ylabel("Empirical Power")
        ax.set_ylim(0, 1)
        ax.set_title(f"{exp} (alpha={alpha})")
        plt.tight_layout()
        plt.savefig(os.path.join(result_dir, f"empirical_power_{exp}_a{alpha}.png"))
        plt.close()

        # Plot FDR for this experiment
        plt.bar(["FDR"], [fdr_value])
        plt.ylim(0, 1)
        plt.ylabel("FDR")
        plt.title(f"{exp} (alpha={alpha})")
        plt.tight_layout()
        plt.savefig(os.path.join(result_dir, f"fdr_{exp}_a{alpha}.png"))
        plt.close()

        for _, row in power_df.iterrows():
            summary_rows.append({
                "experiment": exp,
                "pathway": row["pathway"],
                "empirical_power": row["empirical_power"],
                "FDR": fdr_value,
            })

        print(power_df)
        print(f"{exp} FDR: {fdr_value:.4f}")

    # Generate trend plots grouped by scenario
    for scen, info in scenario_data.items():
        vals = sorted(info["vals"], key=lambda x: x[0])
        params = [v[0] for v in vals]
        labels = [f"{info['label']} = {v}" for v in params]
        power_traces = {g: [] for g in true_groups}
        fdr_list = []
        for val, p_df, fdr in vals:
            for g in true_groups:
                row = p_df.loc[p_df.pathway == g, "empirical_power"]
                power_traces[g].append(row.iloc[0] if not row.empty else 0)
            fdr_list.append(fdr)

        x = np.arange(len(params))
        width = 0.8 / len(true_groups)
        fig, ax = plt.subplots()
        for i, (g, pvals) in enumerate(power_traces.items()):
            offset = (i - (len(true_groups) - 1) / 2) * width
            ax.bar(x + offset, pvals, width, label=g)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_ylabel("Empirical Power")
        ax.set_title(f"{scen} (alpha={alpha})")
        ax.set_ylim(0, 1)
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(result_dir, f"trend_power_{scen}_a{alpha}.png"))
        plt.close()

        fig, ax = plt.subplots()
        ax.bar(x, fdr_list, width=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_ylabel("FDR")
        ax.set_title(f"{scen} (alpha={alpha})")
        ax.set_ylim(0, 1)
        plt.tight_layout()
        plt.savefig(os.path.join(result_dir, f"trend_fdr_{scen}_a{alpha}.png"))
        plt.close()

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(
        os.path.join(result_dir, f"power_fdr_summary_a{alpha}.csv"), index=False
    )


def main():
    result_dir = input("Enter experiment result directory: ").strip()
    if not result_dir:
        print("No directory provided.")
        return
    if not os.path.isdir(result_dir):
        print(f"Directory '{result_dir}' does not exist.")
        return
    for a in (0.05, 0.1):
        compute_metrics(result_dir, alpha=a)


if __name__ == "__main__":
    main()
