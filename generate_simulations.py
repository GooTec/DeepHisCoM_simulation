import os
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.preprocessing import StandardScaler


def prepare_features(data_path: str):
    """Load and preprocess metabolite/clinical dataset."""
    df = pd.read_csv(data_path, index_col=0)
    metabolite = df.iloc[:, 14:]
    clinical = df.iloc[:, :14]

    eps = 1e-6
    X_log = np.log(metabolite.values + eps)
    X_scaled = StandardScaler().fit_transform(X_log)
    df_pre = pd.DataFrame(X_scaled, index=metabolite.index, columns=metabolite.columns)

    mapping = {
        "map00400": ["Phe", "Trp", "Tyr"],
        "map00860": ["Glu", "Gly", "Thr"],
    }

    X1 = df_pre[mapping["map00400"]].values
    X2 = df_pre[mapping["map00860"]].values

    return clinical.reset_index(drop=True), X1, X2


def run_simulation(sim_num: int, clinical: pd.DataFrame, X1: np.ndarray, X2: np.ndarray, rng: np.random.Generator, base_dir: str = "simulation"):
    """Generate outcomes for one simulation number."""
    sim_dir = os.path.join(base_dir, str(sim_num))
    os.makedirs(sim_dir, exist_ok=True)

    # Scenario 1: Linear effects
    w_linear = 1.5
    betas_linear = [0.1, 0.2, 0.3, 0.4]
    for beta in betas_linear:
        eta = beta * (w_linear * X1.sum(axis=1) + w_linear * X2.sum(axis=1))
        pi = expit(eta)
        y_col = f"y_linear_beta_{beta}"
        out_df = pd.concat(
            [clinical, pd.Series(rng.binomial(1, pi), name=y_col)], axis=1
        )
        out_df.to_csv(os.path.join(sim_dir, f"linear_beta_{beta}.csv"), index=False)

    # Scenario 2: Interaction effects
    beta_inter = 0.4
    interaction_ws = [0.1, 0.2, 0.3, 0.4]
    for w_int in interaction_ws:
        inter1 = w_int * (
            X1[:, 0] * X1[:, 1] + X1[:, 0] * X1[:, 2] + X1[:, 1] * X1[:, 2]
        )
        inter2 = w_int * (
            X2[:, 0] * X2[:, 1] + X2[:, 0] * X2[:, 2] + X2[:, 1] * X2[:, 2]
        )
        term1 = inter1 + X1.sum(axis=1)
        term2 = inter2 + X2.sum(axis=1)
        eta = beta_inter * (term1 + term2)
        pi = expit(eta)
        y_col = f"y_inter_w_{w_int}"
        out_df = pd.concat(
            [clinical, pd.Series(rng.binomial(1, pi), name=y_col)], axis=1
        )
        out_df.to_csv(os.path.join(sim_dir, f"interaction_w_{w_int}.csv"), index=False)

    # Scenario 3: Quadratic effects
    betas_quad = [0.6, 0.7, 0.8, 0.9]
    w_quad = np.array([(-1) ** (i + 1) for i in range(X1.shape[1])])
    for beta in betas_quad:
        term1 = (w_quad * (X1 ** 2)).sum(axis=1)
        term2 = (w_quad * (X2 ** 2)).sum(axis=1)
        eta = beta * (term1 + term2)
        pi = expit(eta)
        y_col = f"y_quad_beta_{beta}"
        out_df = pd.concat(
            [clinical, pd.Series(rng.binomial(1, pi), name=y_col)], axis=1
        )
        out_df.to_csv(os.path.join(sim_dir, f"quadratic_beta_{beta}.csv"), index=False)


def main(data_path: str = "./181_metabolite_clinical.csv", n_sim: int = 100):
    os.makedirs("simulation", exist_ok=True)
    clinical, X1, X2 = prepare_features(data_path)
    for sim_num in range(1, n_sim + 1):
        rng = np.random.default_rng(sim_num)
        run_simulation(sim_num, clinical, X1, X2, rng)
    print(f"Saved {n_sim} simulations to 'simulation/' directory.")


if __name__ == "__main__":
    main()
