import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


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
        Significance threshold for declaring discoveries.

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

    for exp, tables in sorted(pval_by_exp.items()):
        df_all = pd.concat(tables, ignore_index=True)
        df_all["reject"] = df_all["pvalue"] <= alpha

        power_df = (
            df_all[df_all["pathway"].isin(true_groups)]
            .groupby("pathway")["reject"]
            .mean()
            .reset_index(name="empirical_power")
        )

        total_rejects = int(df_all["reject"].sum())
        false_rejects = df_all[~df_all["pathway"].isin(true_groups) & df_all["reject"]]
        fdr = len(false_rejects) / total_rejects if total_rejects else 0.0

        # Plot empirical power for this experiment
        ax = power_df.plot(kind="bar", x="pathway", y="empirical_power", legend=False)
        ax.set_xlabel("Pathway")
        ax.set_ylabel("Empirical Power")
        ax.set_ylim(0, 1)
        ax.set_title(f"{exp} (alpha={alpha})")
        plt.tight_layout()
        plt.savefig(os.path.join(result_dir, f"empirical_power_{exp}.png"))
        plt.close()

        # Plot FDR for this experiment
        plt.bar(["FDR"], [fdr])
        plt.ylim(0, 1)
        plt.ylabel("FDR")
        plt.title(f"{exp} (alpha={alpha})")
        plt.tight_layout()
        plt.savefig(os.path.join(result_dir, f"fdr_{exp}.png"))
        plt.close()

        for _, row in power_df.iterrows():
            summary_rows.append({
                "experiment": exp,
                "pathway": row["pathway"],
                "empirical_power": row["empirical_power"],
                "FDR": fdr,
            })

        print(power_df)
        print(f"{exp} FDR: {fdr:.4f}")

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(result_dir, "power_fdr_summary.csv"), index=False)


def main():
    result_dir = input("Enter experiment result directory: ").strip()
    if not result_dir:
        print("No directory provided.")
        return
    if not os.path.isdir(result_dir):
        print(f"Directory '{result_dir}' does not exist.")
        return
    compute_metrics(result_dir)


if __name__ == "__main__":
    main()
