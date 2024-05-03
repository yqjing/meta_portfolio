# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd
from typing import Text, Union
import copy

from tqdm import tqdm

from ...utils import get_or_create_path
from ...log import get_module_logger
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .pytorch_utils import count_parameters
from ...model.base import Model
from ...data.dataset import DatasetH
from ...data.dataset.handler import DataHandlerLP
from ...contrib.model.pytorch_lstm import LSTMModel
from ...contrib.model.pytorch_gru import GRUModel
import pickle5


class RSR(Model):
    """GATs Model

    Parameters
    ----------
    lr : float
        learning rate
    d_feat : int
        input dimensions for each time step
    metric : str
        the evaluation metric used in early stop
    optimizer : str
        optimizer name
    GPU : int
        the GPU ID used for training
    """

    def __init__(
            self,
            stock_index='./data/csi300_stock_index.npy',
            stock2stock_matrix='./data/csi300_multi_stock2stock_all.npy',
            d_feat=6,
            hidden_size=64,
            num_layers=2,
            dropout=0.0,
            n_epochs=200,
            lr=0.001,
            metric="",
            early_stop=20,
            loss="mse",
            base_model="GRU",
            model_path=None,
            optimizer="adam",
            GPU=0,
            seed=None,
            **kwargs
    ):
        # Set logger.
        self.logger = get_module_logger("RSR")
        self.logger.info("RSR pytorch version...")

        # set hyper-parameters.
        self.d_feat = d_feat
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.n_epochs = n_epochs
        self.lr = lr
        self.metric = metric
        self.early_stop = early_stop
        self.optimizer = optimizer.lower()
        self.loss = loss
        self.base_model = base_model
        self.model_path = model_path
        self.device = torch.device("cuda:%d" % (GPU) if torch.cuda.is_available() and GPU >= 0 else "cpu")
        self.seed = seed

        self.stock_index = np.load(stock_index, allow_pickle=True).item()
        stock2stock_matrix = np.load(stock2stock_matrix)
        if len(stock2stock_matrix.shape) == 2:
            stock2stock_matrix = np.expand_dims(stock2stock_matrix, axis=2)
        self.stock2stock_matrix = torch.Tensor(stock2stock_matrix).to(self.device)
        self.num_relation = stock2stock_matrix.shape[2]  # the number of relations
        # with open(market_value_path, "rb") as fh:
        #     # load market value
        #     df_market_value = pickle5.load(fh)
        # # df_market_value = pd.read_pickle(args.market_value_path)
        # self.df_market_value = df_market_value / 1000000000
        # # market value of every day from 07 to 20
        self.batch_size = -1

        self.logger.info(
            "RSR parameters setting:"
            "\nd_feat : {}"
            "\nhidden_size : {}"
            "\nnum_layers : {}"
            "\ndropout : {}"
            "\nn_epochs : {}"
            "\nlr : {}"
            "\nmetric : {}"
            "\nearly_stop : {}"
            "\noptimizer : {}"
            "\nloss_type : {}"
            "\nbase_model : {}"
            "\nmodel_path : {}"
            "\ndevice : {}"
            "\nuse_GPU : {}"
            "\nseed : {}".format(
                d_feat,
                hidden_size,
                num_layers,
                dropout,
                n_epochs,
                lr,
                metric,
                early_stop,
                optimizer.lower(),
                loss,
                base_model,
                model_path,
                self.device,
                self.use_gpu,
                seed,
            )
        )

        if self.seed is not None:
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)

        self.model = RSRModel(
            d_feat=self.d_feat,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            base_model=self.base_model,
            num_relation=self.num_relation
        )
        self.logger.info("model:\n{:}".format(self.model))
        self.logger.info("model size: {:.4f} MB".format(count_parameters(self.model)))

        if optimizer.lower() == "adam":
            self.train_optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        elif optimizer.lower() == "gd":
            self.train_optimizer = optim.SGD(self.model.parameters(), lr=self.lr)
        else:
            raise NotImplementedError("optimizer {} is not supported!".format(optimizer))

        self.fitted = False
        self.model.to(self.device)

    @property
    def use_gpu(self):
        return self.device != torch.device("cpu")

    def mse(self, pred, label):
        loss = (pred - label) ** 2
        return torch.mean(loss)

    def loss_fn(self, pred, label):
        mask = ~torch.isnan(label)

        if self.loss == "mse":
            return self.mse(pred[mask], label[mask])

        raise ValueError("unknown loss `%s`" % self.loss)

    def metric_fn(self, pred, label):

        mask = torch.isfinite(label)

        if self.metric == "ic":
            x = pred[mask]
            y = label[mask]

            vx = x - torch.mean(x)
            vy = y - torch.mean(y)
            return torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx**2)) * torch.sqrt(torch.sum(vy**2)))

        if self.metric == ("", "loss"):
            return -self.loss_fn(pred[mask], label[mask])

        raise ValueError("unknown metric `%s`" % self.metric)

    def get_daily_inter(self, df, shuffle=False):
        # organize the train data into daily batches
        daily_count = df.groupby(level=0).size().values
        daily_index = np.roll(np.cumsum(daily_count), 1)
        daily_index[0] = 0
        if shuffle:
            # shuffle data
            daily_shuffle = list(zip(daily_index, daily_count))
            np.random.shuffle(daily_shuffle)
            daily_index, daily_count = zip(*daily_shuffle)
        return daily_index, daily_count

    def train_epoch(self, train_loader):
        """
        train epoch function
        :param epoch: number of epoch of training
        :param model: model that will be used
        :param optimizer:
        :param train_loader:
        :param writer:
        :param args:
        :param stock2concept_matrix:
        :return:
        """
        self.model.train()

        for i, slc in train_loader.iter_batch():
            feature, label, stock_index, _ = train_loader.get(slc)
            pred = self.model(feature, self.stock2stock_matrix[stock_index][:, stock_index])
            loss = self.loss_fn(pred, label)
            self.train_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.model.parameters(), 3.)
            self.train_optimizer.step()

    def test_epoch(self, test_loader, prefix='Test'):
        """
        :return: loss -> mse
                 scores -> ic
                 rank_ic
                 precision, recall -> precision, recall @1, 3, 5, 10, 20, 30, 50, 100
        """
        self.model.eval()

        scores = []
        losses = []

        for i, slc in test_loader.iter_daily():
            feature, label, stock_index, index = test_loader.get(slc)

            with torch.no_grad():
                pred = self.model(feature, self.stock2stock_matrix[stock_index][:, stock_index])
                loss = self.loss_fn(pred, label)
                losses.append(loss.item())

                if self.metric == 'ic':
                    score = self.metric_fn(pred, label)
                else:
                    score = -loss
                scores.append(score.item())
        return np.mean(losses), np.mean(scores)

    def create_dataloader(self, dataset, segment='train'):
        df = dataset.prepare(segment, col_set=["feature", "label"], data_key=DataHandlerLP.DK_L, )

        start_index = 0
        # slc = slice(pd.Timestamp(dataset.segments[segment][0]), pd.Timestamp(dataset.segments[segment][1]))
        # df['market_value'] = self.df_market_value[slc]
        # df['market_value'] = df['market_value'].fillna(df['market_value'].mean())
        df['stock_index'] = 733
        df['stock_index'] = df.index.get_level_values('instrument').map(self.stock_index).fillna(733).astype(int)
        # the market value and stock_index added to each line

        return DataLoader(df["feature"], df["label"],
                          df['stock_index'],
                          batch_size=self.batch_size, pin_memory=True, start_index=start_index,
                          device=self.device)

    def fit(
            self,
            dataset: DatasetH,
            evals_result=dict(),
            save_path=None,
    ):
        train_loader, valid_loader = self.create_dataloader(dataset, 'train'), self.create_dataloader(dataset, 'valid')
        test_loader = self.create_dataloader(dataset, 'test')

        save_path = get_or_create_path(save_path)
        stop_steps = 0
        best_score = -np.inf
        best_epoch = 0
        evals_result["train"] = []
        evals_result["valid"] = []

        # train
        self.logger.info("training...")
        self.fitted = True

        for step in range(self.n_epochs):
            self.logger.info("Epoch%d:", step)
            self.logger.info("training...")
            self.train_epoch(train_loader)
            self.logger.info("evaluating...")
            # train_loss, train_score = self.test_epoch(train_loader, 'train')
            val_loss, val_score = self.test_epoch(valid_loader, 'valid')
            test_loss, test_score = self.test_epoch(test_loader, 'test')
            self.logger.info("valid %.6f\t test %.6f" % (val_score, test_score))
            # evals_result["train"].append(train_score)
            evals_result["valid"].append(val_score)

            if val_score > best_score:
                best_score = val_score
                stop_steps = 0
                best_epoch = step
                best_param = copy.deepcopy(self.model.state_dict())
            else:
                stop_steps += 1
                if stop_steps >= self.early_stop:
                    self.logger.info("early stop")
                    break

        self.logger.info("best score: %.6lf @ %d" % (best_score, best_epoch))
        self.model.load_state_dict(best_param)
        torch.save(best_param, save_path)

        if self.use_gpu:
            torch.cuda.empty_cache()

    def predict(self, dataset: DatasetH, segment: Union[Text, slice] = "test"):
        if not self.fitted:
            raise ValueError("model is not fitted yet!")

        data_loader = self.create_dataloader(dataset, 'test')
        self.model.eval()

        preds = []
        for i, slc in data_loader.iter_daily():
            feature, label, stock_index, index = data_loader.get(slc)
            with torch.no_grad():
                pred = self.model(feature, self.stock2stock_matrix[stock_index][:, stock_index])
                preds.append(pd.DataFrame({'score': pred.cpu().numpy()}, index=index))

        preds = pd.concat(preds, axis=0)
        return preds


def metric_fn(preds):
    preds = preds[~np.isnan(preds['label'])]
    precision = {}
    recall = {}

    temp = preds.groupby(level='datetime').apply(lambda x: x.sort_values(by='score', ascending=False))
    if len(temp.index[0]) > 2:
        temp = temp.reset_index(level=0).drop('datetime', axis=1)

    for k in [1, 3, 5, 10, 20, 30, 50, 100]:
        precision[k] = temp.groupby(level='datetime').apply(lambda x: (x.label[:k] > 0).sum() / k).mean()
        recall[k] = temp.groupby(level='datetime').apply(lambda x: (x.label[:k] > 0).sum() / (x.label > 0).sum()).mean()

    ic = preds.groupby(level='datetime').apply(lambda x: x.label.corr(x.score)).mean()
    icir = ic / preds.groupby(level='datetime').apply(lambda x: x.label.corr(x.score)).std()
    rank_ic = preds.groupby(level='datetime').apply(lambda x: x.label.corr(x.score, method='spearman')).mean()
    rank_icir = rank_ic / preds.groupby(level='datetime').apply(
        lambda x: x.label.corr(x.score, method='spearman')).std()
    return precision, recall, ic, rank_ic, icir, rank_icir


class DataLoader:

    def __init__(self, df_feature, df_label, df_stock_index, batch_size=800, pin_memory=True,
                 start_index=0, device=None):

        assert len(df_feature) == len(df_label)

        self.df_feature = df_feature.values
        self.df_label = df_label.values
        # self.df_market_value = df_market_value
        self.df_stock_index = df_stock_index
        self.device = device

        if pin_memory:
            self.df_feature = torch.tensor(self.df_feature, dtype=torch.float, device=device)
            self.df_label = torch.tensor(self.df_label, dtype=torch.float, device=device)
            self.df_stock_index = torch.tensor(self.df_stock_index, dtype=torch.long, device=device)

        self.index = df_label.index

        self.batch_size = batch_size
        self.pin_memory = pin_memory
        self.start_index = start_index

        self.daily_count = df_label.groupby(level=0).size().values
        self.daily_index = np.roll(np.cumsum(self.daily_count), 1)
        self.daily_index[0] = 0

    @property
    def batch_length(self):

        if self.batch_size <= 0:
            # this is the default situation
            return self.daily_length

        return len(self.df_label) // self.batch_size

    @property
    def daily_length(self):
        """
        :return:  number of days in the dataloader
        """
        return len(self.daily_count)

    def iter_batch(self):
        if self.batch_size <= 0:
            yield from self.iter_daily_shuffle()
            return

        indices = np.arange(len(self.df_label))
        np.random.shuffle(indices)

        for i in range(len(indices))[::self.batch_size]:
            if len(indices) - i < self.batch_size:
                break
            yield i, indices[i:i + self.batch_size]  # NOTE: advanced indexing will cause copy

    def iter_daily_shuffle(self):
        indices = np.arange(len(self.daily_count))
        np.random.shuffle(indices)
        for i in indices:
            yield i, slice(self.daily_index[i], self.daily_index[i] + self.daily_count[i])

    def iter_daily(self):
        """
        : yield an index and a slice, that from the day
        """
        indices = np.arange(len(self.daily_count))
        for i in indices:
            yield i, slice(self.daily_index[i], self.daily_index[i] + self.daily_count[i])
        # for idx, count in zip(self.daily_index, self.daily_count):
        #     yield slice(idx, idx + count) # NOTE: slice index will not cause copy

    def get(self, slc):
        outs = self.df_feature[slc], self.df_label[slc][:, 0], self.df_stock_index[slc]
        # outs = self.df_feature[slc], self.df_label[slc]

        if not self.pin_memory:
            outs = tuple(torch.tensor(x, device=self.device) for x in outs)

        return outs + (self.index[slc],)


class RSRModel(nn.Module):
    def __init__(self, num_relation, d_feat=6, hidden_size=64, num_layers=2, dropout=0.0, base_model="GRU"):
        super().__init__()

        self.d_feat = d_feat
        self.hidden_size = hidden_size

        self.rnn = nn.GRU(
            input_size=d_feat,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.W = torch.nn.Parameter(torch.randn((hidden_size * 2) + num_relation, 1))
        self.W.requires_grad = True
        torch.nn.init.xavier_uniform_(self.W)
        self.b = torch.nn.Parameter(torch.tensor(0, dtype=torch.float))
        self.b.requires_grad = True
        # Fan in and fan out can not be computed for tensor with fewer than 2 dimensions
        # torch.nn.init.xavier_uniform_(self.b)
        self.leaky_relu = nn.LeakyReLU()
        # if use for loop in forward, change dim to 0
        self.softmax1 = torch.nn.Softmax(dim=1)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x, relation_matrix):
        # N = the number of stock in current slice
        # F = feature length
        # T = number of days, usually = 60, since F*T should be 360
        # x is the feature of all stocks in one day
        x_hidden = x.reshape(len(x), self.d_feat, -1)  # [N, F, T]
        x_hidden = x_hidden.permute(0, 2, 1)  # [N, T, F]
        x_hidden, _ = self.rnn(x_hidden)
        x_hidden = x_hidden[:, -1, :]  # [N, 64]
        # get the last layer embeddings
        # update embedding using relation_matrix
        # relation matrix shape [N, N]
        ei = x_hidden.unsqueeze(1).repeat(1, relation_matrix.shape[0], 1)  # shape N,N,64
        hidden_batch = x_hidden.unsqueeze(0).repeat(relation_matrix.shape[0], 1, 1)  # shape N,N,64
        matrix = torch.cat((ei, hidden_batch, relation_matrix), 2)  # matrix shape N,N,64+relation_number
        weight = (torch.matmul(matrix, self.W) + self.b).squeeze(2)  # weight shape N,N
        weight = self.leaky_relu(weight)  # relu layer
        index = torch.t(torch.nonzero(torch.sum(relation_matrix, 2)))
        mask = torch.zeros(relation_matrix.shape[0], relation_matrix.shape[1], device=x_hidden.device)
        mask[index[0], index[1]] = 1
        valid_weight = mask * weight
        valid_weight = self.softmax1(valid_weight)
        hidden = torch.matmul(valid_weight, x_hidden)
        # hidden = torch.cat((x_hidden, hidden), 1)
        # now hidden shape (N,64) stores all new embeddings
        pred_all = self.fc(hidden + x_hidden).squeeze()
        return pred_all
