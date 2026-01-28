# Environment: PyTorch 2.7.1 (CPU)
import pandas as pd
import numpy as np
import copy
import random
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Data loading
#The hemisphere dataset is provided in the `data/` directory
data = pd.read_excel("data/hemisphere_melting_temperature.xlsx")


X = data.iloc[:,data.columns!= "T (℃)"]
y = data.iloc[:,data.columns == "T (℃)"]


# Normalization processing
x_scaler = StandardScaler()
X_scaled = x_scaler.fit_transform(X)

y_scaler = StandardScaler()
y_scaled = y_scaler.fit_transform(y)


# Data division
X_tr, X_temp, y_tr, y_temp = train_test_split(X_scaled, y_scaled, test_size=0.3, random_state=99)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=99)


X_tr = torch.FloatTensor(X_tr)
X_val = torch.FloatTensor(X_val)
X_test = torch.FloatTensor(X_test)
y_tr = torch.FloatTensor(y_tr)
y_val = torch.FloatTensor(y_val)
y_test = torch.FloatTensor(y_test)


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


# Model training:
set_seed(188)  
model = MLP(input_dim=X_tr.shape[1])
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

patience = 50   
best_val_loss = float("inf")
best_model_state = None
epochs_no_improve = 0


val_losses = []
val_r2s = []

for epoch in range(1, 1001):
    # ===== training =====
    model.train()
    pred = model(X_tr)
    loss = criterion(pred, y_tr)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # ===== Validation =====
    model.eval()
    with torch.no_grad():
        val_pred = model(X_val)
        val_pred_rescaled = y_scaler.inverse_transform(val_pred.numpy())
        y_val_rescaled = y_scaler.inverse_transform(y_val.numpy())

        val_loss = mean_squared_error(y_val_rescaled, val_pred_rescaled)
        val_r2 = r2_score(y_val_rescaled, val_pred_rescaled)
    
    val_losses.append(val_loss)
    val_r2s.append(val_r2)
        
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_state = copy.deepcopy(model.state_dict())
        epochs_no_improve = 0  
    else:
        epochs_no_improve += 1

    print(f"Epoch {epoch}: MSE={val_loss:.2f}, R²={val_r2:.3f}")

    # ===== Early stop =====
    if epochs_no_improve >= patience:
        print(f"Early stop mechanism triggered: At {epoch} epoch , there was no improvement in the validation set")
        break
model.load_state_dict(best_model_state)
print(f"Training completed, best validation set MSE={best_val_loss:.2f}")



# Test set evaluation
model.eval()  
with torch.no_grad(): 
    test_pred = model(X_test)  

    test_pred_rescaled = y_scaler.inverse_transform(test_pred.numpy())
    y_test_rescaled = y_scaler.inverse_transform(y_test.numpy())

    test_mse = mean_squared_error(y_test_rescaled, test_pred_rescaled)
    test_r2 = r2_score(y_test_rescaled, test_pred_rescaled)

print(f"\n[Test Set Evaluation] MSE: {test_mse:.2f}, R²: {test_r2:.3f}")




