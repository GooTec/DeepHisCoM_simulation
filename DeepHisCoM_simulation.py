import os
import random
import argparse
from glob import glob
from collections import OrderedDict

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def load_mapping(path: str) -> dict:
    """Load metabolite-to-pathway mapping from a simple text file.

    Each line should contain a pathway id and a metabolite name separated by
    whitespace or a comma. Lines beginning with '#' are ignored.
    Returns a dictionary {pathway: [metabolites,...]}.
    """
    mapping = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            line = line.replace(',', ' ')
            parts = [p for p in line.split() if p]
            if len(parts) < 2:
                continue
            pathway, metabolite = parts[0], parts[1]
            mapping.setdefault(pathway, []).append(metabolite)
    return mapping


class CustomDataset(Dataset):
    def __init__(self, data: np.ndarray):
        self.x_data = data[:, :-1]
        self.y_data = data[:, -1].reshape(-1, 1)

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        return self.x_data[idx], self.y_data[idx]


# ---------------------------------------------------------------------------
# Model components
# ---------------------------------------------------------------------------

act_fn_by_name = {"tanh": nn.Tanh, "relu": nn.ReLU, "identity": nn.Identity, "leakyrelu": nn.LeakyReLU}

def init_weights(m: nn.Module) -> None:
    if isinstance(m, nn.Linear):
        m.weight.data.fill_(0.01)

class PathwayBlock(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, layer_num: int, act_fn: nn.Module, dropout: nn.Module):
        super().__init__()
        layers = []
        if hidden_dim == 0:
            layers += [nn.Linear(input_dim, 1, bias=False), act_fn, dropout]
        else:
            layers += [nn.Linear(input_dim, hidden_dim, bias=False), act_fn, dropout]
            for _ in range(layer_num - 1):
                layers += [nn.Linear(hidden_dim, hidden_dim, bias=False), act_fn, dropout]
            layers += [nn.Linear(hidden_dim, 1, bias=False), act_fn, dropout]
        self.block = nn.Sequential(*layers)
        self.block.apply(init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DeepHisCoM(nn.Module):
    def __init__(self, nvar, width, layer, covariate, act_fn, dropout_rate):
        super().__init__()
        self.nvar = nvar
        self.covariate = covariate
        dropout = nn.Dropout(dropout_rate)
        self.pathway_nn = nn.ModuleList([
            PathwayBlock(nvar[i], width[i], layer[i], act_fn, dropout) for i in range(len(nvar))
        ])
        self.bn_path = nn.BatchNorm1d(len(nvar))
        self.dropout = dropout
        self.fc_path_disease = nn.Linear(len(nvar) + covariate, 1)
        self.fc_path_disease.weight.data.fill_(0)
        self.fc_path_disease.bias.data.fill_(0.001)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        splits = [0]
        for n in self.nvar:
            splits.append(splits[-1] + n)
        splits.append(splits[-1] + self.covariate)
        path = [self.pathway_nn[i](x[:, splits[i]:splits[i+1]]) for i in range(len(self.nvar))]
        pathway_layer = torch.cat(path, dim=1)
        pathway_layer = self.bn_path(pathway_layer)
        pathway_layer = pathway_layer / torch.norm(pathway_layer, 2)
        x_cat = torch.cat([pathway_layer, x[:, splits[len(self.nvar)]:splits[len(self.nvar)+1]]], dim=1)
        x_cat = self.dropout(x_cat)
        x_cat = self.fc_path_disease(x_cat)
        return torch.sigmoid(x_cat)


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser("DeepHisCoM permutation simulation")
    parser.add_argument("--dir", type=str, default=".", help="working directory")
    parser.add_argument("--simulation_dir", type=str, default="simulation", help="directory containing simulation folders")
    parser.add_argument("--start_sim", type=int, default=1, help="start simulation number")
    parser.add_argument("--end_sim", type=int, help="end simulation number (inclusive)")
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--perm", type=int, default=1000)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--activation", type=str, default="leakyrelu")
    parser.add_argument("--loss", type=str, default="BCELoss")
    parser.add_argument("--reg_type", type=str, default="l1")
    parser.add_argument("--reg_const_pathway_disease", type=float, default=0)
    parser.add_argument("--reg_const_bio_pathway", type=float, default=0)
    parser.add_argument("--leakyrelu_const", type=float, default=0.2)
    parser.add_argument("--dropout_rate", type=float, default=0.5)
    parser.add_argument("--experiment_name", type=str, default="exp")
    parser.add_argument("--batch_size", type=int, default=0)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--stop_type", type=int, default=0)
    parser.add_argument("--divide_rate", type=float, default=0.2)
    parser.add_argument("--count_lim", type=int, default=5)
    parser.add_argument("--cov", type=int, default=0)
    parser.add_argument("--mapping_file", type=str, default="metabolite_mapping.set")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    act_fn = act_fn_by_name[args.activation]
    if args.activation == "leakyrelu":
        act_fn = act_fn(args.leakyrelu_const)
    else:
        act_fn = act_fn()

    os.chdir(args.dir)
    os.makedirs(os.path.join(args.experiment_name, "tmp"), exist_ok=True)

    # load data
    df_meta = pd.read_csv("181_metabolite_clinical.csv", index_col=0)
    metabolite = df_meta.iloc[:, 14:]

    mapping = load_mapping(args.mapping_file)
    group_to_feats = {}
    for group, mets in mapping.items():
        valid = [m for m in mets if m in metabolite.columns]
        if valid:
            group_to_feats[group] = valid

    annot_rows = []
    for group, mets in group_to_feats.items():
        for m in mets:
            annot_rows.append({"metabolite": m, "group": group})
    annot = pd.DataFrame(annot_rows)

    met_columns = annot["metabolite"].unique().tolist()
    metabolite = metabolite[met_columns]

    sim_df = pd.read_csv(args.scenario)
    out_col = [c for c in sim_df.columns if c.startswith("y_")][0]
    sim_df = sim_df[[out_col]].rename(columns={out_col: "phenotype"})
    train_base = pd.concat([metabolite.reset_index(drop=True), sim_df], axis=1)

    eps = 1e-6
    X_log = np.log(metabolite.values + eps)
    X_scaled = StandardScaler().fit_transform(X_log)
    df_scaled = pd.DataFrame(X_scaled, columns=metabolite.columns)
    train_base.loc[:, metabolite.columns] = df_scaled

    groupunique = list(OrderedDict.fromkeys(annot["group"]))
    nvar = [sum(annot["group"] == g) for g in groupunique]

    feature_cols = []
    for g in groupunique:
        feature_cols.extend(annot[annot["group"] == g]["metabolite"].tolist())
    train_base = train_base[feature_cols + ["phenotype"]]

    node_num = pd.read_csv("layerinfo.csv")["node_num"].tolist()
    layer_num = pd.read_csv("layerinfo.csv")["layer_num"].tolist()

    cov_num = 0
    if args.cov:
        cov_df = pd.read_csv("cov.csv")
        cov_num = len(cov_df["x"])

    eps = 1e-6
    X_log = np.log(metabolite.values + eps)
    X_scaled = StandardScaler().fit_transform(X_log)
    metabolite_scaled = pd.DataFrame(X_scaled, columns=metabolite.columns)

    start_sim = args.start_sim
    end_sim = args.end_sim if args.end_sim is not None else start_sim

    for sim_num in range(start_sim, end_sim + 1):
        sim_dir = os.path.join(args.simulation_dir, str(sim_num))
        for scenario_path in sorted(glob(os.path.join(sim_dir, "*.csv"))):
            experiment = os.path.splitext(os.path.basename(scenario_path))[0]
            sim_df = pd.read_csv(scenario_path)
            out_col = [c for c in sim_df.columns if c.startswith("y_")][0]
            sim_df = sim_df[[out_col]].rename(columns={out_col: "phenotype"})
            train_base = pd.concat([metabolite_scaled.reset_index(drop=True), sim_df], axis=1)
            train_base = train_base[feature_cols + ["phenotype"]]

            for permutation in range(args.perm):
                torch.manual_seed(permutation)
                random.seed(permutation)
                np.random.seed(permutation)
                train = train_base.copy()
                if permutation != 0:
                    train["phenotype"] = np.random.permutation(train["phenotype"])
                ph = train["phenotype"]
                train = (train - train.mean()) / train.std()
                train["phenotype"] = ph

                tensor = torch.from_numpy(train.values).float()
                dataset = CustomDataset(tensor)

                if 3 <= args.stop_type <= 4:
                    t_idx, v_idx, _, _ = train_test_split(
                        range(len(dataset)), dataset.y_data, stratify=dataset.y_data, test_size=args.divide_rate
                    )
                    train_split = Subset(dataset, t_idx)
                    test_split = Subset(dataset, v_idx)
                    if args.batch_size == 0:
                        train_loader = DataLoader(train_split, batch_size=len(train_split), shuffle=True)
                        test_loader = DataLoader(test_split, batch_size=len(test_split))
                    else:
                        train_loader = DataLoader(train_split, batch_size=args.batch_size, shuffle=True)
                        test_loader = DataLoader(test_split, batch_size=args.batch_size)
                else:
                    if args.batch_size == 0:
                        train_loader = DataLoader(dataset, batch_size=len(dataset), shuffle=True)
                    else:
                        train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
                    test_loader = None

                model = DeepHisCoM(nvar, node_num, layer_num, cov_num, act_fn, args.dropout_rate).to(device)
                optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
                criterion = nn.BCELoss() if args.loss.lower() == "bceloss" else nn.MSELoss()

                count = 0
                best_param = None
                scores = []

                for epoch in range(10000):
                    batch_scores = []
                    for x_batch, y_batch in train_loader:
                        x_batch = x_batch.to(device)
                        y_batch = y_batch.to(device)
                        optimizer.zero_grad()
                        output = torch.squeeze(model(x_batch))
                        y_batch = torch.squeeze(y_batch)
                        loss = criterion(output, y_batch)
                        if args.reg_const_pathway_disease != 0:
                            for param in model.fc_path_disease.parameters():
                                if args.reg_type == "l1":
                                    loss = loss + args.reg_const_pathway_disease * torch.norm(param, 1)
                                else:
                                    loss = loss + args.reg_const_pathway_disease * torch.norm(param, 2) ** 2
                        if args.reg_const_bio_pathway != 0:
                            for param in model.pathway_nn.parameters():
                                if args.reg_type == "l1":
                                    loss = loss + args.reg_const_bio_pathway * torch.norm(param, 1)
                                else:
                                    loss = loss + args.reg_const_bio_pathway * torch.norm(param, 2) ** 2
                        loss.backward()
                        optimizer.step()
                        if args.stop_type == 1:
                            batch_scores.append(-loss.item())
                        if args.stop_type == 2:
                            if torch.sum(torch.isnan(output)) == 0:
                                auc = roc_auc_score(y_batch.cpu().detach().numpy(), output.cpu().detach().numpy())
                            else:
                                auc = 0
                            batch_scores.append(auc)
                        if args.stop_type == 5:
                            best_param = model.fc_path_disease.weight.detach().cpu().numpy()[0]
                    if test_loader is not None:
                        for x_test, y_test in test_loader:
                            x_test = x_test.to(device)
                            y_test = y_test.to(device)
                            output = torch.squeeze(model(x_test))
                            y_test = torch.squeeze(y_test)
                            if args.stop_type == 3:
                                loss_test = criterion(output, y_test).item()
                                batch_scores.append(-loss_test)
                            if args.stop_type == 4:
                                if torch.sum(torch.isnan(output)) == 0:
                                    auc_test = roc_auc_score(y_test.cpu().detach().numpy(), output.cpu().detach().numpy())
                                else:
                                    auc_test = 0
                                batch_scores.append(auc_test)
                    if args.stop_type in [1, 2, 3, 4]:
                        avg_score = sum(batch_scores) / len(batch_scores)
                        scores.append(avg_score)
                        if avg_score >= max(scores):
                            count = 0
                            best_param = model.fc_path_disease.weight.detach().cpu().numpy()[0]
                        else:
                            count += 1
                        if count > args.count_lim:
                            break
                    elif args.stop_type == 5:
                        if epoch > args.count_lim:
                            break
                if best_param is not None:
                    out_dir = os.path.join(args.simulation_dir, str(sim_num), experiment, str(permutation))
                    os.makedirs(out_dir, exist_ok=True)
                    np.savetxt(os.path.join(out_dir, "param.txt"), best_param)


if __name__ == "__main__":
    main()
