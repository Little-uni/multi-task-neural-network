import numpy as np
import pandas as pd
import time

import torch
from torch import nn
from torchvision.transforms import transforms
from torch.utils.data import Dataset, DataLoader

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


#%% define functions
class ASHDataset(Dataset):
    def __init__(self, images, labels=None):
        self.X = images
        self.y = labels
         
    def __len__(self):
        return (len(self.X))
    
    def __getitem__(self, i):
        return self.X[i, ].float(), self.y[i, ]

class MultiTaskNeuralNet(nn.Module):
  def __init__(self):
    super(MultiTaskNeuralNet, self).__init__()
    self.net = nn.Sequential(
      nn.Linear(6, 20),
      nn.BatchNorm1d(20),
      nn.ReLU(inplace=True),
      nn.Linear(20, 6),
      nn.BatchNorm1d(6),
      nn.ReLU(inplace=True),
      nn.Linear(6, 17)
      )

  def forward(self, x):
    x = x.view(x.size(0), -1)
    x = self.net(x)
    return x

def post_evaluation(X,Y,scaler_feature,scaler_label,data_feature):
    dataset=ASHDataset(X, Y)
    loader_final = DataLoader(dataset, 10, shuffle=False,drop_last=False)
    final_pred=np.array([])
    final_true=np.array([])
    model.eval()
    for images, labels in loader_final:
      final_output = model(images.to(device)).view(-1)
      final_label = labels.to(device)
      final_pred = np.append(final_pred, final_output.detach().cpu().numpy())
      final_true = np.append(final_true, final_label.detach().cpu().numpy())

    final_pred_recover=scaler_label.inverse_transform(final_pred.reshape(-1, 17))
    final_true_recover=scaler_label.inverse_transform(final_true.reshape(-1, 17))

    metric_r2=r2_score(final_true_recover,final_pred_recover,multioutput='raw_values')
    metric_mape=mean_absolute_error(final_true_recover,final_pred_recover,multioutput='raw_values')/np.mean(final_true_recover,axis=0)
    metric_rmse=np.sqrt(mean_squared_error(final_true_recover,final_pred_recover,multioutput='raw_values'))
    
    return metric_r2, metric_mape, metric_rmse, final_true_recover,final_pred_recover

#%% load and test the processed Polish dataset from Franus et al. with the three models discussed in Fig. 5

data_raw_test=pd.read_excel(r"G:\My Drive\Manuscript\Paper 17_DOE ash\Data\REE dataset_Franus.xlsx",sheet_name=0,index_col=0)
data_name_test=list(data_raw_test) 
data_raw_test.reset_index(drop=True,inplace=True)

import joblib
scaler1 = joblib.load(r"G:\My Drive\Manuscript\Paper 17_DOE ash\Data\3 types of multi-task models for the Polish dataset\scaler1.save")
scaler2 = joblib.load(r"G:\My Drive\Manuscript\Paper 17_DOE ash\Data\3 types of multi-task models for the Polish dataset\scaler2.save")

device = torch.device("cpu")

data_feature_test=data_raw_test.iloc[-12:,np.r_[1,2,3,4,7,8]]
data_label_test=data_raw_test.iloc[-12:,11:]

data_feature_scaled_test=scaler1.transform(data_feature_test)
data_feature_scaled_test=pd.DataFrame(data_feature_scaled_test)

data_label_scaled_test=scaler2.transform(data_label_test)
data_label_scaled_test=pd.DataFrame(data_label_scaled_test)

Y_test=data_label_scaled_test.copy()
X_test=data_feature_scaled_test.copy()

X_test = torch.tensor(X_test.values).to(device)
Y_test = torch.tensor(Y_test.values).to(device)

test_data = ASHDataset(X_test, Y_test)
test_loader = DataLoader(test_data, 10, shuffle=False,drop_last=False)

model = MultiTaskNeuralNet().to(device)

# present model
model.load_state_dict(torch.load(r"G:\My Drive\Manuscript\Paper 17_DOE ash\Data\3 types of multi-task models for the Polish dataset\present model.pt"))
metric_r2_test,metric_mape_test, metric_rmse_test, final_true_recover_test,final_pred_recover_test = \
post_evaluation(X=X_test, Y=Y_test, scaler_feature=scaler1, scaler_label=scaler2, data_feature=data_feature_test)
pred_present_model = final_pred_recover_test[:,0]

# non-transferred model
model.load_state_dict(torch.load(r"G:\My Drive\Manuscript\Paper 17_DOE ash\Data\3 types of multi-task models for the Polish dataset\non-transferred model.pt"))
metric_r2_test,metric_mape_test, metric_rmse_test, final_true_recover_test,final_pred_recover_test = \
post_evaluation(X=X_test, Y=Y_test, scaler_feature=scaler1, scaler_label=scaler2, data_feature=data_feature_test)
pred_nontransferred_model = final_pred_recover_test[:,0]

# transferred model
model.load_state_dict(torch.load(r"G:\My Drive\Manuscript\Paper 17_DOE ash\Data\3 types of multi-task models for the Polish dataset\transferred model.pt"))
metric_r2_test,metric_mape_test, metric_rmse_test, final_true_recover_test,final_pred_recover_test = \
post_evaluation(X=X_test, Y=Y_test, scaler_feature=scaler1, scaler_label=scaler2, data_feature=data_feature_test)
pred_transferred_model = final_pred_recover_test[:,0]