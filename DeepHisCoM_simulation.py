# %% [markdown]
# # DeepHisCoM Simulation Notebook
# 
# 이 노트북은 Python 스크립트 대신 Jupyter Notebook 형태로, `simulation/` 폴더의 시뮬레이션 결과와 대사체 정보를 이용하여 DeepHisCoM 모델을 학습합니다.
# 
# - 사전 조건: `181_metabolite_clinical.csv`, `layerinfo.csv`, `annot.csv`, 그리고 `simulation/` 폴더 내부의 CSV 파일들이 작업 디렉토리에 위치해야 합니다.

# %%
import os
import random
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset

# %%
# 설정 섹션
seed = 100
perm_times = 1000
gpu_num = 0
# 사용할 시나리오 파일 이름
scenario_file = 'linear_beta_0.1.csv'  # 예시: simulation/linear_beta_0.1.csv
# 학습 하이퍼파라미터
batch_size = 0
learning_rate = 0.001
dropout_rate = 0.5
leakyrelu_const = 0.2
activation = 'leakyrelu'
loss_type = 'BCELoss'
reg_type = 'l1'
reg_const_path_disease = 0
reg_const_bio_path = 0
stop_type = 0
divide_rate = 0.2
count_lim = 5
cov = 0

# %%
# 디렉토리 이동 및 데이터 로딩
os.chdir(os.getcwd())

# 대사체 데이터
df_meta = pd.read_csv('181_metabolite_clinical.csv', index_col=0)
metabolite = df_meta.iloc[:, 14:].reset_index(drop=True)

# 시뮬레이션 결과
sim_df = pd.read_csv(os.path.join('simulation', scenario_file))
outcome_col = [c for c in sim_df.columns if c.startswith('y_')][0]
sim_df = sim_df[[outcome_col]].rename(columns={outcome_col: 'phenotype'})

# 통합 데이터프레임 생성
train_base = pd.concat([metabolite, sim_df], axis=1)
train_base.head()

# %%
# 전처리: 로그 변환 + 표준화
eps = 1e-6
X_log = np.log(metabolite.values + eps)
X_scaled = StandardScaler().fit_transform(X_log)
df_pre = pd.DataFrame(X_scaled, columns=metabolite.columns)


# %%
# 경로 매핑 설정
mapping = {
    'map00400': ['Phe', 'Trp', 'Tyr'],
    'map00860': ['Glu', 'Gly', 'Thr']
}
X1 = df_pre[mapping['map00400']].values
X2 = df_pre[mapping['map00860']].values

# %%
# layerinfo 및 annot 로딩
layerinfo = pd.read_csv('layerinfo.csv')
node_num = layerinfo['node_num'].tolist()
layer_num = layerinfo['layer_num'].tolist()
annot = pd.read_csv('annot.csv')

# %%
# 그룹별 변수 수 계산
from collections import OrderedDict
groupunique = list(OrderedDict.fromkeys(annot['group']))
nvar = [(annot['group']==g).sum() for g in groupunique]
cov_num = 0

# %%
# 모델 정의 및 데이터셋 클래스
class CustomDataset(Dataset):
    def __init__(self, DATA, inputlength):
        self.x_data = DATA[:, :-1]
        self.y_data = DATA[:, -1].reshape(-1,1)
    def __len__(self): return len(self.x_data)
    def __getitem__(self, idx): return self.x_data[idx], self.y_data[idx]

act_fn_by_name = {"tanh": nn.Tanh, "relu": nn.ReLU, "identity": nn.Identity, "leakyrelu": nn.LeakyReLU}

def init_weights(m):
    if isinstance(m, nn.Linear): m.weight.data.fill_(0.01)

class pathwayblock(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_num):
        super().__init__()
        layers = []
        if hidden_dim == 0:
            layers += [nn.Linear(input_dim,1,bias=False), act_fn, nn.Dropout(dropout_rate)]
        else:
            layers += [nn.Linear(input_dim,hidden_dim,bias=False), act_fn, nn.Dropout(dropout_rate)]
            for _ in range(layer_num-1): layers += [nn.Linear(hidden_dim,hidden_dim,bias=False), act_fn, nn.Dropout(dropout_rate)]
            layers += [nn.Linear(hidden_dim,1,bias=False), act_fn, nn.Dropout(dropout_rate)]
        self.block = nn.Sequential(*layers)
        self.block.apply(init_weights)
    def forward(self, x): return self.block(x)

class DeepHisCoM(nn.Module):
    def __init__(self, nvar, width, layer, covariate):
        super().__init__()
        self.pathway_nn = nn.ModuleList([pathwayblock(nvar[i], width[i], layer[i]) for i in range(len(nvar))])
        self.bn_path = nn.BatchNorm1d(len(nvar))
        self.fc_path_disease = nn.Linear(len(nvar)+covariate,1)
        self.fc_path_disease.weight.data.fill_(0)
        self.fc_path_disease.bias.data.fill_(0.001)
    def forward(self, x):
        splits = np.cumsum([0]+nvar+[cov_num])
        p = [self.pathway_nn[i](x[:,splits[i]:splits[i+1]]) for i in range(len(nvar))]
        path = torch.cat(p,1)
        path = self.bn_path(path)
        path = path/torch.norm(path,2)
        x_cat = torch.cat([path, x[:,splits[-1]:]],1)
        x_cat = nn.Dropout(dropout_rate)(x_cat)
        return torch.sigmoid(self.fc_path_disease(x_cat))


# %%
# 학습 루프: 단일 permutation 예시
# 전체 반복(permutation)을 수행하려면 for문으로 감싸세요
np.random.seed(seed)
train = train_base.copy()
phen = train['phenotype']
train = (train - train.mean())/train.std()
train['phenotype'] = phen

tensor = torch.from_numpy(train.values).float()
dataset = CustomDataset(tensor, tensor.shape[1])
loader = DataLoader(dataset, batch_size=(batch_size or len(dataset)), shuffle=True)

model = DeepHisCoM(nvar, node_num, layer_num, cov_num)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.BCELoss() if loss_type=='BCELoss' else nn.MSELoss()

# 한 epoch 훈련 예시
model.train()
for x_batch, y_batch in loader:
    optimizer.zero_grad()
    y_pred = model(x_batch)
    loss = criterion(y_pred.squeeze(), y_batch.squeeze())
    loss.backward()
    optimizer.step()
print('Training complete')


