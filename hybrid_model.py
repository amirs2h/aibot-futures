import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import xgboost as xgb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import joblib
import os

# -------------------- LSTM PyTorch Model --------------------
class LSTMNet(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super(LSTMNet, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 3)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

# -------------------- Dataset Class --------------------
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y, sequence_length=24):
        self.X = X
        self.y = y
        self.seq_len = sequence_length

    def __len__(self):
        return len(self.X) - self.seq_len

    def __getitem__(self, idx):
        return (
            torch.tensor(self.X[idx:idx+self.seq_len], dtype=torch.float32),
            torch.tensor(self.y[idx+self.seq_len], dtype=torch.long),
        )

# -------------------- Hybrid Model Trainer --------------------
class HybridTrainer:
    def __init__(self, df, features, target, model_dir="models"):
        self.df = df.copy()
        self.features = features
        self.target = target
        self.sequence_length = 24
        self.scaler = StandardScaler()
        self.xgb_model = None
        self.lstm_model = None
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)

    def prepare_data(self):
        X = self.df[self.features].values
        y = self.df[self.target].values
        X_scaled = self.scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, shuffle=False)
        return X_train, X_test, y_train, y_test

    def train_xgb(self, X_train, y_train):
        model = xgb.XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05)
        model.fit(X_train, y_train)
        joblib.dump(model, os.path.join(self.model_dir, "xgb_model.pkl"))
        self.xgb_model = model
        return model

    def train_lstm(self, X_train, y_train, patience=3):
        dataset = TimeSeriesDataset(X_train, y_train, self.sequence_length)
        loader = DataLoader(dataset, batch_size=64, shuffle=True)

        model = LSTMNet(input_size=X_train.shape[1])
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        model.train()
        best_loss = float('inf')
        epochs_no_improve = 0

        for epoch in range(50):
            epoch_loss = 0.0
            for xb, yb in loader:
                optimizer.zero_grad()
                out = model(xb)
                loss = loss_fn(out, yb)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            epoch_loss /= len(loader)

            print(f"Epoch {epoch+1}, Loss: {epoch_loss:.4f}")
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                torch.save(model.state_dict(), os.path.join(self.model_dir, "lstm_model.pth"))
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

        self.lstm_model = model
        return model

    def predict_proba(self, X_input, xgb_weight=0.5, lstm_weight=0.5):
        X_scaled = self.scaler.transform(X_input)
        xgb_pred = self.xgb_model.predict_proba(X_scaled[self.sequence_length:])

        lstm_dataset = TimeSeriesDataset(X_scaled, np.zeros(len(X_scaled)), self.sequence_length)
        lstm_loader = DataLoader(lstm_dataset, batch_size=64, shuffle=False)
        self.lstm_model.eval()

        lstm_pred = []
        with torch.no_grad():
            for xb, _ in lstm_loader:
                out = self.lstm_model(xb)
                lstm_pred.append(torch.softmax(out, dim=1).numpy())

        lstm_pred = np.vstack(lstm_pred)
        hybrid_pred = xgb_weight * xgb_pred + lstm_weight * lstm_pred
        return hybrid_pred

    def predict_hybrid(self, X_input):
        hybrid_proba = self.predict_proba(X_input)
        final_preds = np.argmax(hybrid_proba, axis=1)
        return final_preds

    def save_scaler(self):
        joblib.dump(self.scaler, os.path.join(self.model_dir, "scaler.pkl"))

    def load_models(self):
        self.scaler = joblib.load(os.path.join(self.model_dir, "scaler.pkl"))
        self.xgb_model = joblib.load(os.path.join(self.model_dir, "xgb_model.pkl"))
        self.lstm_model = LSTMNet(input_size=len(self.features))
        self.lstm_model.load_state_dict(torch.load(os.path.join(self.model_dir, "lstm_model.pth")))
        self.lstm_model.eval()

class HybridModel(HybridTrainer):
    pass
