import numpy as np
import torch
from torch import nn
from xgboost import XGBRegressor


class SequenceModel(nn.Module):
    def __init__(self, kind, features=9, length=12):
        super().__init__()
        self.kind = kind
        if kind == "lstm":
            self.encoder = nn.LSTM(features, 16, batch_first=True)
        elif kind == "transformer":
            self.projection = nn.Linear(features, 16)
            self.position = nn.Parameter(torch.randn(1, length, 16)*.02)
            layer = nn.TransformerEncoderLayer(16, 2, dim_feedforward=32, dropout=0, batch_first=True)
            self.encoder = nn.TransformerEncoder(layer, 1, enable_nested_tensor=False)
        else:
            raise ValueError("Unknown sequence model")
        self.head = nn.Linear(16, 2)

    def forward(self, x):
        if self.kind == "lstm":
            encoded, _ = self.encoder(x)
        else:
            encoded = self.encoder(self.projection(x)+self.position)
        return self.head(encoded[:, -1])


def fit_predict(kind, x_train, y_train, x_test, seed, epochs=18):
    mean, std = x_train.mean(axis=(0, 1)), x_train.std(axis=(0, 1))
    std = np.maximum(std, 1e-6)
    train, test = (x_train-mean)/std, (x_test-mean)/std
    if kind == "xgboost":
        predictions = []
        for target in range(2):
            model = XGBRegressor(n_estimators=80, max_depth=3, learning_rate=.05,
                                 objective="reg:squarederror", n_jobs=1, random_state=seed,
                                 tree_method="hist", reg_lambda=5)
            model.fit(train.reshape(len(train), -1), y_train[:, target])
            predictions.append(model.predict(test.reshape(len(test), -1)))
        prediction = np.column_stack(predictions)
    else:
        torch.manual_seed(seed)
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        model = SequenceModel(kind)
        optimizer = torch.optim.Adam(model.parameters(), lr=.005)
        tx, ty = torch.from_numpy(train), torch.from_numpy(y_train)
        model.train()
        for _ in range(epochs):
            optimizer.zero_grad()
            loss = torch.mean((model(tx)-ty)**2)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            prediction = model(torch.from_numpy(test)).numpy()
    # Fixed sanity bounds, not tuned on test outcomes; convert percentage targets back to returns.
    prediction = prediction.astype(float)/100
    prediction[:, 0] = np.clip(prediction[:, 0], -.04, .04)
    prediction[:, 1] = np.clip(prediction[:, 1], .003, .12)
    return prediction
