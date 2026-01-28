# Environment: PyTorch 2.7.1 (CPU)
import pandas as pd
import numpy as np
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import random
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# Data loading
#The SHTT dataset for model's training is provided in the `data/` directory, except for data from part of the ongoing study
data_SHTT = pd.read_excel("data/SHTT_melting_temperature.xlsx")


X_SHTT = data_SHTT.iloc[:,data_SHTT.columns!= "T (℃)"]
y_SHTT = data_SHTT.iloc[:,data_SHTT.columns == "T (℃)"]


# Normalization processing
x_scaler_SHTT = StandardScaler()
X_scaled_SHTT = x_scaler_SHTT.fit_transform(X_SHTT)

y_scaler_SHTT = StandardScaler()
y_scaled_SHTT = y_scaler_SHTT.fit_transform(y_SHTT)


X_scaled_SHTT = torch.FloatTensor(X_scaled_SHTT)
y_scaled_SHTT = torch.FloatTensor(y_scaled_SHTT)


# Model structure
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, dropout_rate=0.3):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),  
            nn.ReLU(),                         
            nn.BatchNorm1d(hidden_dim),        
            nn.Dropout(dropout_rate),          

            nn.Linear(hidden_dim, hidden_dim), 
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout_rate),
        )
        self.regressor = nn.Linear(hidden_dim, 1)  

    def forward(self, x):
        x = self.feature_extractor(x)
        return self.regressor(x)

    
# Random seed definition
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# Load the pre-trained model on hemisphere melting temperature dataset
model_trans = MLP(input_dim=X_scaled_SHTT.shape[1])
model_trans.load_state_dict(torch.load("pretrained_hemisphere_model.pth"))


# freeze
for param in model_trans.parameters():
    param.requires_grad = False
for param in model_trans.regressor.parameters():
    param.requires_grad = True

optimizer = optim.Adam(
    model_trans.regressor.parameters(), lr=0.001
)


set_seed(188)  
criterion = nn.MSELoss()


patience = 50  
best_train_loss = float("inf")  
best_model_trans_state = None
epochs_no_improve = 0
train_losses = []
train_r2s = []

for epoch in range(1, 1001):
    model_trans.train()
    pred = model_trans(X_scaled_SHTT)
    loss = criterion(pred, y_scaled_SHTT)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    model_trans.eval()
    with torch.no_grad():
        train_pred = model_trans(X_scaled_SHTT)

        
        train_pred_rescaled = y_scaler_SHTT.inverse_transform(train_pred.numpy())
        y_tr_rescaled = y_scaler_SHTT.inverse_transform(y_scaled_SHTT.numpy())

        
        train_loss = mean_squared_error(y_tr_rescaled, train_pred_rescaled)
        train_r2 = r2_score(y_tr_rescaled, train_pred_rescaled)

        
    train_losses.append(train_loss)
    train_r2s.append(train_r2)
    
    
    if train_loss < best_train_loss:
        best_train_loss = train_loss
        best_model_trans_state = copy.deepcopy(model_trans.state_dict())
        epochs_no_improve = 0  
    else:
        epochs_no_improve += 1

    print(f"Epoch {epoch}: MSE={train_loss:.2f}, R²={train_r2:.3f}")

    # ===== Early stop =====
    if epochs_no_improve >= patience:
        print(f"Early stop mechanism triggered: At {epoch} epoch , there was no improvement in the training set")
        break

model_trans.load_state_dict(best_model_trans_state)
print(f"Training completed, best training  set MSE={best_train_loss:.2f}")


# Test set evaluation
# The test data were shown in manuscript, users can load their own test set as a pandas DataFrame with the same columns as X_SHTT
test_file_path = "path/to/test_SHTT_melting_temperature.xlsx"
try:
    data_test = pd.read_excel(test_file_path)
except FileNotFoundError:
    print("Test dataset not found. Skipping test evaluation.")
    data_test = None
if data_test is not None:
    X_test_scaled = x_scaler_SHTT.transform(data_test[X_SHTT.columns])
    X_test_scaled = torch.FloatTensor(X_test_scaled)
    model_trans.eval()
    with torch.no_grad():
        test_pred = model_trans(X_test_scaled)
    test_pred_rescaled = y_scaler_SHTT.inverse_transform(test_pred.numpy())
    print("Test prediction:", test_pred_rescaled)
else:
    print("Test set not provided. Skipping test evaluation.")
