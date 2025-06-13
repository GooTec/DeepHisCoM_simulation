import os
import argparse
from collections import OrderedDict
import numpy as np
import pandas as pd

from DeepHisCoM_simulation import load_mapping


def load_groups(mapping_file: str) -> list:
    """Load pathway groups matching the training script."""
    df_meta = pd.read_csv("181_metabolite_clinical.csv", index_col=0)
    metabolite = df_meta.iloc[:, 14:]
    mapping = load_mapping(mapping_file)
    annot = pd.DataFrame(
        [{"metabolite": m, "group": g} for g, ms in mapping.items() for m in ms if m in metabolite.columns]
    )
    annot_unique = annot.drop_duplicates(subset="metabolite", keep="first")
    return list(OrderedDict.fromkeys(annot_unique["group"]))


def compute_pvalues(base_dir: str, groups: list, n_perm: int = 100) -> None:
    """Compute permutation p-values for a single experiment directory."""
    obs_path = os.path.join(base_dir, "0", "param.txt")
    if not os.path.exists(obs_path):
        return
    obs_param = np.loadtxt(obs_path, ndmin=1)

    perm_params = []
    for p in range(1, n_perm + 1):
        param_path = os.path.join(base_dir, str(p), "param.txt")
        if os.path.exists(param_path):
            perm_params.append(np.loadtxt(param_path, ndmin=1))
    if not perm_params:
        return
    perm_params = np.stack(perm_params, axis=0)

    abs_obs = np.abs(obs_param)
    abs_perm = np.abs(perm_params)

    pvals = []
    for i in range(len(obs_param)):
        greater = np.sum(abs_perm[:, i] >= abs_obs[i])
        pval = (greater + 1) / (len(perm_params) + 1)
        pvals.append(pval)

    names = groups + [f"covariate_{i+1}" for i in range(len(obs_param) - len(groups))]
    df = pd.DataFrame({"param": names[: len(obs_param)], "pvalue": pvals})
    df.to_csv(os.path.join(base_dir, "pvalue.csv"), index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Compute permutation p-values")
    parser.add_argument("--start_sim", type=int, default=1, help="start simulation number")
    parser.add_argument("--end_sim", type=int, help="end simulation number (inclusive)")
    parser.add_argument("--exp_root", type=str, default="exp", help="root directory containing experiments")
    parser.add_argument("--mapping_file", type=str, default="metabolite_mapping.set", help="pathway mapping file")
    parser.add_argument("--n_perm", type=int, default=100, help="number of permutations")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    exp_root = args.exp_root
    if not os.path.isdir(exp_root):
        return

    start_sim = args.start_sim
    end_sim = args.end_sim if args.end_sim is not None else start_sim

    groups = load_groups(args.mapping_file)

    for sim_num in range(start_sim, end_sim + 1):
        sim_dir = os.path.join(exp_root, str(sim_num))
        if not os.path.isdir(sim_dir):
            continue
        for experiment in sorted(os.listdir(sim_dir)):
            base_dir = os.path.join(sim_dir, experiment)
            if os.path.isdir(base_dir):
                compute_pvalues(base_dir, groups, n_perm=args.n_perm)


if __name__ == "__main__":
    main()
