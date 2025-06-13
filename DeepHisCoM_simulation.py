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
    def __len__(self): return len(self.x_data)
    def __getitem__(self, idx): return self.x_data[idx], self.y_data[idx]

# ---------------------------------------------------------------------------
# Model components
# ---------------------------------------------------------------------------

act_fn_by_name = {"tanh": nn.Tanh, "relu": nn.ReLU, "identity": nn.Identity, "leakyrelu": nn.LeakyReLU}

def init_weights(m: nn.Module) -> None:
    if isinstance(m, nn.Linear): nn.init.xavier_uniform_(m.weight)

class PathwayBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_num, act_fn, dropout):
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
    def forward(self, x): return self.block(x)

class DeepHisCoM(nn.Module):
    def __init__(self, nvar, width, layer, covariate, act_fn, dropout_rate):
        super().__init__()
        self.nvar = nvar; self.covariate = covariate
        dropout = nn.Dropout(dropout_rate)
        self.pathway_nn = nn.ModuleList([
            PathwayBlock(nvar[i], width[i], layer[i], act_fn, dropout)
            for i in range(len(nvar))
        ])
        self.bn_path = nn.BatchNorm1d(len(nvar))
        self.dropout = dropout
        self.fc_path_disease = nn.Linear(len(nvar) + covariate, 1)
        self.fc_path_disease.weight.data.zero_(); self.fc_path_disease.bias.data.fill_(0.001)
    def forward(self, x):
        splits = [0]
        for n in self.nvar: splits.append(splits[-1] + n)
        splits.append(splits[-1] + self.covariate)
        path = [self.pathway_nn[i](x[:, splits[i]:splits[i+1]]) for i in range(len(self.nvar))]
        pathway_layer = torch.cat(path, dim=1)
        pathway_layer = self.bn_path(pathway_layer)
        x_cat = torch.cat([pathway_layer, x[:, splits[-2]:splits[-1]]], dim=1)
        x_cat = self.dropout(x_cat)
        return torch.sigmoid(self.fc_path_disease(x_cat))

# ---------------------------------------------------------------------------
# Training per permutation with early stopping
# ---------------------------------------------------------------------------

def train_once(train_loader, val_loader, lr, bs, args, device, nvar, width, layer, cov_num, act_fn):
    model = DeepHisCoM(nvar, width, layer, cov_num, act_fn, args.dropout_rate).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss() if args.loss.lower() == "bceloss" else nn.MSELoss()

    best_auc = -np.inf; best_param = None; patience_cnt = 0
    for epoch in range(1, args.max_epochs + 1):
        model.train(); losses = []
        for x_batch, y_batch in train_loader:
            x, y = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad(); output = model(x).squeeze()
            loss = criterion(output, y.squeeze())
            loss.backward(); optimizer.step(); losses.append(loss.item())
        model.eval(); preds, labels = [], []
        with torch.no_grad():
            for x_val, y_val in val_loader:
                pred = model(x_val.to(device)).squeeze().cpu().numpy()
                preds.extend(pred.tolist()); labels.extend(y_val.squeeze().numpy().tolist())
        val_auc = roc_auc_score(labels, preds)
        
        if val_auc > best_auc:
            best_auc = val_auc; best_param = model.fc_path_disease.weight.detach().cpu().numpy()[0]; patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= args.patience:
                print(f"Early stopping at epoch {epoch}")
                # print(f"Epoch {epoch:03d} | lr={lr:.4f}, bs={bs} | Loss={np.mean(losses):.4f} | Val AUC={val_auc:.4f}")
                break
        
    return best_param, best_auc

# ---------------------------------------------------------------------------
# Main logic with hyperparameter search and permutations
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, default=".")
    parser.add_argument("--simulation_dir", type=str, default="simulation")
    parser.add_argument("--start_sim", type=int, default=1)
    parser.add_argument("--end_sim", type=int)
    parser.add_argument("--perm", type=int, default=100)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--activation", type=str, default="leakyrelu")
    parser.add_argument("--loss", type=str, default="BCELoss")
    parser.add_argument("--reg_const_pathway_disease", type=float, default=0)
    parser.add_argument("--reg_const_bio_pathway", type=float, default=0)
    parser.add_argument("--leakyrelu_const", type=float, default=0.2)
    parser.add_argument("--dropout_rate", type=float, default=0.5)
    parser.add_argument("--experiment_name", type=str, default="exp")
    parser.add_argument("--mapping_file", type=str, default="metabolite_mapping_intersection.set")
    parser.add_argument("--scenario", type=str)
    args = parser.parse_args()

    torch.manual_seed(args.seed); random.seed(args.seed); np.random.seed(args.seed)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    act_cls = act_fn_by_name[args.activation]
    act_fn = act_cls(args.leakyrelu_const) if args.activation == "leakyrelu" else act_cls()

    os.chdir(args.dir); os.makedirs(args.experiment_name, exist_ok=True)

    df_meta = pd.read_csv("181_metabolite_clinical.csv", index_col=0)
    metabolite = df_meta.iloc[:,14:]
    mapping = load_mapping(args.mapping_file)

    annot_rows = []
    for g, ms in mapping.items():
        for m in ms:
            if m in metabolite.columns:          # 데이터에 실제 존재하는 변수만
                annot_rows.append({"metabolite": m, "group": g})

    annot = pd.DataFrame(annot_rows)

    # -----------------------------------------------------------------
    # ❷ 중복-대사체 제거 + pathway 정리
    #     ↳ 한 대사체가 여러 그룹에 있을 경우, "첫 번째 등장"에만 남김
    # -----------------------------------------------------------------
    seen   = set()
    groups = []          # pathway 순서
    cols_by_group = []   # pathway별 고유 column 리스트

    for g in annot["group"].unique():            # 입력 파일 순서 유지
        cols = [m for m in annot.loc[annot.group == g, "metabolite"]
                if m not in seen]
        if len(cols) == 0:                       # 실제 변수 0개인 pathway는 건너뜀
            continue
        seen.update(cols)
        groups.append(g)
        cols_by_group.append(cols)

    # -----------------------------------------------------------------
    # ❸ 최종 feature 열 재배열
    # -----------------------------------------------------------------
    ordered_cols = [c for sub in cols_by_group for c in sub]
    metabolite   = metabolite[ordered_cols]      # <<— 새 순서 적용
    nvar         = [len(sub) for sub in cols_by_group]

    assert sum(nvar) == metabolite.shape[1], \
        f"nvar 합({sum(nvar)}) != feature 개수({metabolite.shape[1]}) — 매핑 파일 점검 필요"

    # -----------------------------------------------------------------
    # ❹ log 변환·스케일링 이후 동일
    # -----------------------------------------------------------------
    eps = 1e-6
    X_log    = np.log(metabolite.values + eps)
    X_scaled = StandardScaler().fit_transform(X_log)
    metabolite_scaled = pd.DataFrame(X_scaled, columns=metabolite.columns)

    # layerinfo.csv 길이 맞추기
    li    = pd.read_csv("layerinfo.csv")
    width = li["node_num"].tolist() [:len(nvar)]
    layer = li["layer_num"].tolist()[:len(nvar)]
    assert len(width) == len(layer) == len(nvar), \
        "layerinfo.csv 행 수가 pathway 수와 다릅니다."
    cov_num=0;  
    if os.path.exists("cov.csv"): cov_num=len(pd.read_csv("cov.csv"))

    sims = [args.scenario] if args.scenario else []
    if not sims:
        end = args.end_sim or args.start_sim
        for sim in range(args.start_sim, end+1):
            sims += sorted(glob(os.path.join(args.simulation_dir, str(sim), "*.csv")))

    lr_list = [0.01]
    bs_list = [16, 32, 64]

    for path in sims:
        sim_num = os.path.basename(os.path.dirname(path))
        exp = os.path.splitext(os.path.basename(path))[0]
        df_s = pd.read_csv(path)
        ycol = [c for c in df_s if c.startswith("y_")][0]
        phen = df_s[[ycol]].rename(columns={ycol: "phenotype"})
        train_base = pd.concat([metabolite_scaled.reset_index(drop=True), phen], axis=1)
        train_base = train_base[list(metabolite_scaled.columns) + ["phenotype"]]

        # Hyperparameter search on original data (perm=0)
        best_val_search = -np.inf; best_lr = None; best_bs = None
        df0 = train_base.copy()
        # Normalize only features for hyperparameter search
        df0_feat = df0[metabolite_scaled.columns.tolist()]
        df0_feat_norm = (df0_feat - df0_feat.mean()) / df0_feat.std()
        df0_norm = pd.concat([df0_feat_norm, df0['phenotype']], axis=1)
        base_tensor = torch.from_numpy(df0_norm.values).float()
        ds0 = CustomDataset(base_tensor)
        for lr in lr_list:
            for bs in bs_list:
                idx_t, idx_v, _, _ = train_test_split(
                    range(len(ds0)), ds0.y_data, stratify=ds0.y_data, test_size=0.2)
                tl = DataLoader(Subset(ds0, idx_t), batch_size=bs, shuffle=True)
                vl = DataLoader(Subset(ds0, idx_v), batch_size=bs)
                _, val_auc = train_once(tl, vl, lr, bs, args, device, nvar, width, layer, cov_num, act_fn)
                if val_auc > best_val_search:
                    best_val_search = val_auc; best_lr = lr; best_bs = bs
        print(f"Best hyperparams for sim {sim_num}, exp {exp}: lr={best_lr}, bs={best_bs}, AUC={best_val_search:.4f}")

        # Permutations including original
        for perm in range(args.perm):
            print(f"Sim {sim_num}, perm {perm}")
            np.random.seed(perm); torch.manual_seed(perm); random.seed(perm)
            df = train_base.copy()
            if perm != 0:
                df['phenotype'] = np.random.permutation(df['phenotype'])
            # Normalize only features for permutation
            df_feat = df[metabolite_scaled.columns.tolist()]
            df_feat_norm = (df_feat - df_feat.mean()) / df_feat.std()
            df_norm = pd.concat([df_feat_norm, df['phenotype']], axis=1)
            tensor = torch.from_numpy(df_norm.values).float()
            ds = CustomDataset(tensor)
            idx_t, idx_v, _, _ = train_test_split(
                range(len(ds)), ds.y_data, stratify=ds.y_data, test_size=0.2)
            tl = DataLoader(Subset(ds, idx_t), batch_size=best_bs, shuffle=True)
            vl = DataLoader(Subset(ds, idx_v), batch_size=best_bs)
            param, auc = train_once(tl, vl, best_lr, best_bs, args, device, nvar, width, layer, cov_num, act_fn)
            outd = os.path.join(args.experiment_name, sim_num, exp, str(perm))
            os.makedirs(outd, exist_ok=True)
            np.savetxt(os.path.join(outd, "param.txt"), param)
            if perm == 0:
                print(f"Original AUC: {auc:.4f}")

if __name__ == "__main__": main()
