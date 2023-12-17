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

def train(model, train_loader, test_loader, loss_func, loss_weight, opt, num_epochs=10, writer=None):
  all_training_loss = np.zeros((0,2))
  all_test_loss = np.zeros((0,2))
  
  training_step = 0
  training_loss = 0.0
  print_every = 500
  
  start = time.process_time()
  
  for i in range(num_epochs):
    epoch_start = time.process_time() 
    model.train()
    for images, labels in train_loader:
      training_step+=10
      images, labels = images.to(device), labels.to(device)
      opt.zero_grad()

      preds = model(images).double()
      labels=labels.view(-1, len(loss_weight))
      
      loss = []
      for i_label in range(len(loss_weight)):
          loss.append(loss_func(preds[:,i_label], labels[:,i_label])*loss_weight[i_label])      
      loss=sum(loss)/sum(loss_weight)   
      loss.backward()
      opt.step()
      
      training_loss += loss.item()
      if training_step/10 % print_every == 0:
        training_loss /= print_every  
        all_training_loss = np.concatenate((all_training_loss, [[training_step, training_loss]]))

        print('  Epoch %d @ step %d: Train Loss: %3f' % (i, training_step, training_loss))
        training_loss = 0.0


    model.eval()
    with torch.no_grad():
      validation_loss = 0.0
      count = 0
      for images, labels in test_loader:
        count += 1
        images, labels = images.to(device), labels.to(device)
        
        outputs = model(images).double()
        labels=labels.view(-1, len(loss_weight))
          
        loss = []
        for i_label in range(len(loss_weight)):
            loss.append(loss_func(outputs[:,i_label], labels[:,i_label])*loss_weight[i_label])   
        loss=sum(loss)/sum(loss_weight)  
        validation_loss+=loss

      validation_loss/=count
      all_test_loss = np.concatenate((all_test_loss, [[training_step, validation_loss]]))
      
      epoch_time = time.process_time() - epoch_start
      print('Epoch %d Test Loss: %3f,  time: %.1fs' % (i, validation_loss, epoch_time))


  total_time = time.process_time() - start
  print('Final Test Loss: %3f, Total time: %.1fs' % (validation_loss, total_time))

  return {'Loss': { 'Train loss': all_training_loss, 'Test loss': all_test_loss }}

class SingleTaskNeuralNet(nn.Module):
  def __init__(self):
    super(SingleTaskNeuralNet, self).__init__()
    self.net = nn.Sequential(
      nn.Linear(len(data_feature_scaled.columns), 20),
      nn.BatchNorm1d(20),
      nn.ReLU(inplace=True),
      nn.Linear(20, 6),
      nn.BatchNorm1d(6),
      nn.ReLU(inplace=True),
      nn.Linear(6, 1)
      )

  def forward(self, x):
    x = x.view(x.size(0), -1)
    x = self.net(x)
    return x

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

    final_pred_recover=scaler_label.inverse_transform(final_pred.reshape(-1, len(data_label.columns)))
    final_true_recover=scaler_label.inverse_transform(final_true.reshape(-1, len(data_label.columns)))

    metric_r2=r2_score(final_true_recover,final_pred_recover,multioutput='raw_values')
    metric_mape=mean_absolute_error(final_true_recover,final_pred_recover,multioutput='raw_values')/np.mean(final_true_recover,axis=0)
    metric_rmse=np.sqrt(mean_squared_error(final_true_recover,final_pred_recover,multioutput='raw_values'))
    
    return metric_r2, metric_mape, metric_rmse, final_true_recover,final_pred_recover

#%% load, preprocess, and prepare the data

data_raw=pd.read_excel(r"G:\My Drive\Manuscript\Paper 17_DOE ash\Data\REE dataset_10% sample.xlsx",sheet_name=0,index_col=0)
data_name=list(data_raw)
data_raw.reset_index(drop=True,inplace=True)

accuracy_sum=pd.DataFrame([])

device = torch.device("cpu")
#%% Single-task neural net model

data_feature=data_raw.iloc[:,np.r_[1,2,3,4,7,8]]
data_label=data_raw.iloc[:,[11]]

# scale features
scaler1 = preprocessing.StandardScaler()
data_feature_scaled=scaler1.fit_transform(data_feature)
data_feature_scaled=pd.DataFrame(data_feature_scaled)

# scale labels
scaler2 = preprocessing.StandardScaler()
data_label_scaled=scaler2.fit_transform(data_label)
data_label_scaled=pd.DataFrame(data_label_scaled)

accuracy_auto=pd.DataFrame([])
for i in range(1):
    Y_train, Y_test=train_test_split(data_label_scaled, test_size = 0.2, random_state = i)
    Y_train.sort_index(inplace=True)
    Y_test.sort_index(inplace=True)
    X_train=data_feature_scaled.loc[Y_train.index]
    X_test=data_feature_scaled.loc[Y_test.index]

    X_train = torch.tensor(X_train.values).to(device)
    Y_train = torch.tensor(Y_train.values).to(device)
    X_test = torch.tensor(X_test.values).to(device)
    Y_test = torch.tensor(Y_test.values).to(device)

    train_data = ASHDataset(X_train, Y_train)
    test_data = ASHDataset(X_test, Y_test)
    train_loader = DataLoader(train_data, 10, shuffle=True,drop_last=False)
    test_loader = DataLoader(test_data, 10, shuffle=False,drop_last=False)

    model = SingleTaskNeuralNet().to(device)

    # reinitialize model parameters
    for layer in model.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()

    loss = nn.MSELoss()
    loss_weight = torch.ones(1)

    optimizer = torch.optim.Adam(model.parameters(),lr=1e-3,weight_decay=1e-2)

    train(model, train_loader, test_loader, loss, loss_weight, optimizer, num_epochs=100)
    
    metric_r2_train,metric_mape_train, metric_rmse_train, final_true_recover_train,final_pred_recover_train = \
    post_evaluation(X=X_train, Y=Y_train, scaler_feature=scaler1, scaler_label=scaler2, data_feature=data_feature)
    metric_r2_test,metric_mape_test, metric_rmse_test, final_true_recover_test,final_pred_recover_test = \
    post_evaluation(X=X_test, Y=Y_test, scaler_feature=scaler1, scaler_label=scaler2, data_feature=data_feature)
    
    accuracy_ind= pd.DataFrame([metric_r2_train[0], metric_mape_train[0], metric_rmse_train[0], metric_r2_test[0], metric_mape_test[0], metric_rmse_test[0]], index = ['r2_train', 'mape_train','rmse_train','r2_test','mape_test','rmse_test'])
    accuracy_auto=pd.concat([accuracy_auto, accuracy_ind.transpose()],axis=0)

accuracy_sum=pd.concat([accuracy_sum, pd.concat([accuracy_auto.mean(axis=0),accuracy_auto.std(axis=0)],axis=1,ignore_index=True).transpose()],axis=0)

#%% Multi-task neural net model, optimal loss weights

data_feature=data_raw.iloc[:,np.r_[1,2,3,4,7,8]]
data_label=data_raw.iloc[:,11:]

# scale features
scaler1 = preprocessing.StandardScaler()
data_feature_scaled=scaler1.fit_transform(data_feature)
data_feature_scaled=pd.DataFrame(data_feature_scaled)

# scale labels
scaler2 = preprocessing.StandardScaler()
data_label_scaled=scaler2.fit_transform(data_label)
data_label_scaled=pd.DataFrame(data_label_scaled)

accuracy_auto=pd.DataFrame([])
for i in range(1):
    Y_train, Y_test=train_test_split(data_label_scaled, test_size = 0.2, random_state = i)
    Y_train.sort_index(inplace=True)
    Y_test.sort_index(inplace=True)
    X_train=data_feature_scaled.loc[Y_train.index]
    X_test=data_feature_scaled.loc[Y_test.index]

    X_train = torch.tensor(X_train.values).to(device)
    Y_train = torch.tensor(Y_train.values).to(device)
    X_test = torch.tensor(X_test.values).to(device)
    Y_test = torch.tensor(Y_test.values).to(device)

    train_data = ASHDataset(X_train, Y_train)
    test_data = ASHDataset(X_test, Y_test)
    train_loader = DataLoader(train_data, 10, shuffle=True,drop_last=False)
    test_loader = DataLoader(test_data, 10, shuffle=False,drop_last=False)

    device = torch.device("cpu")
    model = MultiTaskNeuralNet().to(device)

    # reinitialize model parameters
    for layer in model.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()

    loss = nn.MSELoss()
    loss_weight = torch.tensor(data_label.mean(axis=0).values)

    optimizer = torch.optim.Adam(model.parameters(),lr=1e-3,weight_decay=1e-2)

    train(model, train_loader, test_loader, loss, loss_weight, optimizer, num_epochs=100)
    
    metric_r2_train,metric_mape_train, metric_rmse_train, final_true_recover_train,final_pred_recover_train = \
    post_evaluation(X=X_train, Y=Y_train, scaler_feature=scaler1, scaler_label=scaler2, data_feature=data_feature)
    metric_r2_test,metric_mape_test, metric_rmse_test, final_true_recover_test,final_pred_recover_test = \
    post_evaluation(X=X_test, Y=Y_test, scaler_feature=scaler1, scaler_label=scaler2, data_feature=data_feature)
    
    accuracy_ind= pd.DataFrame([metric_r2_train[0], metric_mape_train[0], metric_rmse_train[0], metric_r2_test[0], metric_mape_test[0], metric_rmse_test[0]], index = ['r2_train', 'mape_train','rmse_train','r2_test','mape_test','rmse_test'])
    accuracy_auto=pd.concat([accuracy_auto, accuracy_ind.transpose()],axis=0)

accuracy_sum=pd.concat([accuracy_sum, pd.concat([accuracy_auto.mean(axis=0),accuracy_auto.std(axis=0)],axis=1,ignore_index=True).transpose()],axis=0)

#%% final comparision
accuracy_sum.index = ['Single-task model_mean', 'Single-task model_std','Multi-task model_mean', 'Multi-task model_std']
print(accuracy_sum)
