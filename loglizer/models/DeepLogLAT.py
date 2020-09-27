import logging
from collections import defaultdict
import torch
import pandas as pd
from sklearn.metrics import (accuracy_score, f1_score,
                             recall_score, precision_score)

logger = logging.getLogger(__name__)


class DeepLogLAT(torch.nn.Module):
    def __init__(self, n_io, n_hidden, n_layers, bidirectional, device, topk):
        super().__init__()
        self._n_io = n_io
        topology = (n_io, n_hidden, n_layers)
        self._lstm = torch.nn.LSTM(
            *topology, batch_first=True, bidirectional=bidirectional)
        self._topk = topk
        self._device = device
        self._criterion = torch.nn.CrossEntropyLoss()
        multiplier = 2 if bidirectional else 1
        self._linear = torch.nn.Linear(n_hidden * multiplier, n_io)

    def forward(self, input_dict):
        y = input_dict["window_y"].long().view(-1).to(self._device)
        batch_size = y.size()[0]
        x = input_dict["x"].view(batch_size, -1)
        x = torch.nn.functional.one_hot(x, num_classes=self._n_io)
        outputs, _ = self._lstm(x.to(self._device).float())
        logits = self._linear(outputs[:, -1, :])
        y_pred = logits.softmax(dim=-1)
        loss = self._criterion(y_pred, y)
        return_dict = {'loss': loss, 'y_pred': y_pred}
        return return_dict

    def fit(self, train_loader, epochs=10):
        self.to(self._device)
        model = self.train()
        optimizer = torch.optim.Adam(model.parameters())
        for epoch in range(1, epochs + 1):
            batch_cnt = 0
            epoch_loss = 0
            for batch_input in train_loader:
                loss = model.forward(batch_input)["loss"]
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                epoch_loss += loss.item()
                batch_cnt += 1
            epoch_loss = epoch_loss / batch_cnt
            logger.info(
                "Epoch {}/{}, training loss: {:.5f}".format(
                    epoch, epochs, epoch_loss))

    def evaluate(self, test_loader):
        self.eval()  # set to evaluation mode
        with torch.no_grad():
            y_pred = []
            store_dict = defaultdict(list)
            for batch_input in test_loader:
                return_dict = self.forward(batch_input)
                y_pred = return_dict["y_pred"]

                store_dict["SessionId"].extend(
                    batch_input["SessionId"].data.cpu().numpy().reshape(-1))
                store_dict["y"].extend(
                    batch_input["y"].data.cpu().numpy().reshape(-1))
                store_dict["window_y"].extend(
                    batch_input["window_y"].data.cpu().numpy().reshape(-1))
                window_prob, window_pred = torch.max(y_pred, 1)
                store_dict["window_pred"].extend(
                    window_pred.data.cpu().numpy().reshape(-1))
                store_dict["window_prob"].extend(
                    window_prob.data.cpu().numpy().reshape(-1))
                top_indice = torch.topk(y_pred, self._topk)[1]  # b x topk
                store_dict["topk_indice"].extend(top_indice.data.cpu().numpy())

            window_pred = store_dict["window_pred"]
            window_y = store_dict["window_y"]

            store_df = pd.DataFrame(store_dict)
            store_df["anomaly"] = store_df.apply(
                lambda x: x["window_y"] not in x["topk_indice"], axis=1
            ).astype(int)

            store_df.drop(["window_pred", "window_y"], axis=1)
            store_df = store_df.groupby('SessionId', as_index=False).sum()
            store_df["anomaly"] = (store_df["anomaly"] > 0).astype(int)
            store_df["y"] = (store_df["y"] > 0).astype(int)
            y_pred = store_df["anomaly"]
            y_true = store_df["y"]

            metrics = {"window_acc": accuracy_score(window_y, window_pred),
                       "session_acc": accuracy_score(y_true, y_pred),
                       "f1": f1_score(y_true, y_pred),
                       "recall": recall_score(y_true, y_pred),
                       "precision": precision_score(y_true, y_pred)}
            logger.info([(k, round(v, 5)) for k, v in metrics.items()])
            return metrics
