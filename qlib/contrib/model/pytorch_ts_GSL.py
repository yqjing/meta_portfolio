# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd
import copy
from qlib.utils import get_or_create_path
from qlib import get_module_logger

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Sampler
from qlib.data.dataset import DatasetH
from qlib.contrib.model.pytorch_utils import count_parameters

from qlib.model.base import Model
from qlib.data.dataset.handler import DataHandlerLP
from pytorch_graph_model_dgl import HomographModel, HeterographModel


import dgl
import dgl.nn as dglnn
import dgl.function as fn
from dgl.nn.pytorch import edge_softmax
from dgl.nn.pytorch.utils import Identity


class Graphs(Model):
    """Graphs Model
    Parameters
    ----------
    lr : float
        learning rate
    d_feat : int
        input dimensions for each time step
    metric : str
    metric : str
    metric : str
        the evaluation metric used in early stop
    optimizer : str
        optimizer name
    GPU : int
        the GPU ID used for training
    """

    def __init__(
        self,
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
        graph_model="GAT",
        model_path=None,  # TODO: use pretrained model
        optimizer="adam",
        GPU=0,
        n_jobs=10,
        seed=None,
        rel_encoding=None,
        use_corr_en=False,
        corr_en=None,
        stock_name_list=None,
        logger=None,
        generate_weight=False,
        **kwargs,
    ):
        super().__init__()
        # Set logger.
        self.logger = logger
        self.logger.info(f"{graph_model}s pytorch version...")

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
        self.graph_model = graph_model
        self.model_path = model_path
        self.device = torch.device(
            "cuda:%d" % (GPU) if torch.cuda.is_available() and GPU >= 0 else "cpu"
        )
        self.seed = seed
        self.n_jobs = n_jobs
        self.stock_name_list = stock_name_list

        self.logger.info(
            "{}s parameters setting:"
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
                graph_model,
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

        if self.graph_model == "GAT":
            self.model = GATModel(
                d_feat=self.d_feat,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,  # RNN layer
                dropout=self.dropout,
                base_model=self.base_model,
            )
        elif self.graph_model == "simpleHGN":
            self.model = SimpleHeteroHGN(
                d_feat=self.d_feat,
                edge_dim=self.hidden_size,
                num_etypes=rel_encoding.shape[-1],
                num_hidden=self.hidden_size,
                num_layers=self.num_layers,  # RNN layer
                dropout=self.dropout,
                feat_drop=0.5,
                attn_drop=0.5,
                negative_slope=0.05,
                residual=True,
                alpha=0.05,
                base_model=self.base_model,
                num_graph_layer=2,
                heads=[8] * self.num_layers,
            )
        elif self.graph_model == "RSR":  # TODO
            pass
        elif self.graph_model == "GSLGraphModel":
            self.model = GSLGraphModel(
                d_feat=self.d_feat,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                dropout=self.dropout,
                base_model=self.base_model,
                generate_weight=generate_weight
                # num_graph_layer默认都为2, 看二跳邻居的信息
            )

        self.logger.info("model:\n{:}".format(self.graph_model))
        self.logger.info("model size: {:.4f} MB".format(count_parameters(self.model)))

        if optimizer.lower() == "adam":
            self.train_optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        elif optimizer.lower() == "gd":
            self.train_optimizer = optim.SGD(self.model.parameters(), lr=self.lr)
        else:
            raise NotImplementedError(
                "optimizer {} is not supported!".format(optimizer)
            )

        self.fitted = False
        self.model.to(self.device)
        self.model.set_graph(rel_encoding=rel_encoding, device=self.device)

        self.rel_encoding = rel_encoding

        self.corr_en = corr_en
        self.use_corr_en = use_corr_en

    @property
    def use_gpu(self):
        return self.device != torch.device("cpu")

    def mse(self, pred, label):
        loss = (pred - label) ** 2
        return torch.mean(loss)

    def loss_fn(self, pred, label, alpha=0.1):
        mask = ~torch.isnan(label)

        if self.loss == "mse":
            return self.mse(pred[mask], label[mask])
        elif self.loss == "rank_mse":
            mse_loss = self.mse(pred[mask], label[mask])
            # print(f"prediction的shape为{pred.shape}")

            batch_size = pred.shape[0]
            all_one = torch.ones(batch_size, 1, dtype=torch.float32).to(self.device)
            pre_pw_dif = torch.sub(
                torch.reshape(pred, (-1, 1)) @ all_one.t(),
                all_one @ torch.reshape(pred, (1, -1)),
            )
            gt_pw_dif = torch.sub(
                all_one @ torch.reshape(label, (1, -1)),
                torch.reshape(label, (-1, 1)) @ all_one.t(),
            )
            mask = mask.type(torch.float32)
            mask_pw = mask @ mask.t()
            rank_loss = torch.mean(F.relu(pre_pw_dif * gt_pw_dif * mask_pw))
            return mse_loss + alpha * rank_loss

        raise ValueError("unknown loss `%s`" % self.loss)

    def metric_fn(self, pred, label):
        mask = torch.isfinite(label)

        if self.metric in ("", "loss"):
            return -self.loss_fn(pred[mask], label[mask])

        raise ValueError("unknown metric `%s`" % self.metric)

    def cal_my_IC(self, prediction, label):
        pd_label = pd.Series(label.detach().cpu().numpy().squeeze())
        pd_prediction = pd.Series(prediction.detach().cpu().numpy().squeeze())
        return pd_prediction.corr(pd_label)

    def get_daily_inter(self, df, corr_en=None, shuffle=False):
        # organize the train data into daily batches
        daily_count = df.groupby(level=0).size().values
        daily_index = np.roll(np.cumsum(daily_count), 1)
        daily_index[0] = 0

        stocks_daily_ = []
        for date, stocks in pd.Series(index=df.index, dtype=np.float32).groupby(
            "datetime"
        ):
            stocks_daily_.append(list(stocks.loc[date, :].index))

        indexes = np.array([i for i in range(len(daily_index))])
        if shuffle:
            np.random.shuffle(indexes)
        daily_index = daily_index[indexes]
        daily_count = daily_count[indexes]
        stocks_daily = [stocks_daily_[i] for i in indexes]
        if self.use_corr_en:
            corr_en_shuffled = corr_en[indexes]
            return daily_index, daily_count, stocks_daily, corr_en_shuffled
        return daily_index, daily_count, stocks_daily

    def get_mixed_graph(
        self, corr_A, rel_encoding, higher_threshold=0.98, lower_threshold=-1
    ):
        if len(rel_encoding.shape) == 3:
            rel_encoding = rel_encoding.sum(axis=-1)
        stock_num = rel_encoding.shape[0]
        predefined = rel_encoding.astype(np.bool_)
        mask1 = corr_A > lower_threshold
        new_graph = np.logical_and(mask1, predefined)
        mask2 = corr_A > higher_threshold
        new_graph = np.logical_or(mask2, new_graph)
        new_graph = np.logical_or(new_graph, np.eye(stock_num).astype(np.bool_))
        # 转换为无向图
        new_graph = np.logical_or(new_graph, new_graph.T)
        return new_graph.astype(np.float64)

    def train_epoch(self, x_train, y_train):
        self.model.train()
        x_train_values = x_train.values
        y_train_values = np.squeeze(y_train.values)

        if self.use_corr_en:
            (
                daily_index,
                daily_count,
                stocks_daily,
                corr_en_shuffled,
            ) = self.get_daily_inter(
                x_train, corr_en=self.corr_en["train"], shuffle=True
            )
        else:
            daily_index, daily_count, stocks_daily = self.get_daily_inter(
                x_train, shuffle=True
            )

        for i in range(len(daily_index)):
            idx, count, stks = daily_index[i], daily_count[i], stocks_daily[i]
            if self.use_corr_en:
                corr_A = corr_en_shuffled[i]
                A = self.get_mixed_graph(corr_A, self.rel_encoding)
                self.model.set_graph(rel_encoding=A, device=self.device)
            index = [self.stock_name_list.index(stk) for stk in stks]
            batch = slice(idx, idx + count)

            feature = torch.from_numpy(x_train_values[batch]).float().to(self.device)
            label = torch.from_numpy(y_train_values[batch]).float().to(self.device)

            pred = self.model(feature.float(), index)
            loss = self.loss_fn(pred, label)

            self.train_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.model.parameters(), 3.0)
            self.train_optimizer.step()

    def test_epoch(self, data_x, data_y, name="valid"):
        # prepare training data
        x_values = data_x.values
        y_values = np.squeeze(data_y.values)
        self.model.eval()

        scores = []
        losses = []
        my_ICs = []
        # organize the test data into daily batches
        if self.use_corr_en:
            (
                daily_index,
                daily_count,
                stocks_daily,
                corr_en_shuffled,
            ) = self.get_daily_inter(data_x, corr_en=self.corr_en[name], shuffle=False)
        else:
            daily_index, daily_count, stocks_daily = self.get_daily_inter(
                data_x, shuffle=False
            )

        for i in range(len(daily_index)):
            idx, count, stks = daily_index[i], daily_count[i], stocks_daily[i]
            if self.use_corr_en:
                corr_A = corr_en_shuffled[i]
                A = self.get_mixed_graph(corr_A, self.rel_encoding)
                self.model.set_graph(rel_encoding=A, device=self.device)
            index = [self.stock_name_list.index(stk) for stk in stks]
            batch = slice(idx, idx + count)
            feature = torch.from_numpy(x_values[batch]).float().to(self.device)
            label = torch.from_numpy(y_values[batch]).float().to(self.device)

            with torch.no_grad():
                pred = self.model(feature.float(), index)
            loss = self.loss_fn(pred, label)
            losses.append(loss.item())
            IC = self.cal_my_IC(pred, label)
            my_ICs.append(IC)

            score = self.metric_fn(pred, label)
            scores.append(score.item())
        return np.nanmean(losses), np.nanmean(scores), np.nanmean(my_ICs)

    def fit(
        self,
        dataset: DatasetH,
        evals_result=dict(),
        save_path=None,
    ):
        df_train, df_valid = dataset.prepare(
            ["train", "valid"],
            col_set=["feature", "label"],
            data_key=DataHandlerLP.DK_L,
        )

        df_test = dataset.prepare(
            "test", col_set=["feature", "label"], data_key=DataHandlerLP.DK_I
        )

        if df_train.empty or df_valid.empty:
            raise ValueError(
                "Empty data from dataset, please check your dataset config."
            )

        x_train, y_train = df_train["feature"], df_train["label"]
        x_valid, y_valid = df_valid["feature"], df_valid["label"]
        x_test, y_test = df_test["feature"], df_test["label"]

        save_path = get_or_create_path(save_path)

        stop_steps = 0
        train_loss = 0
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
            self.train_epoch(x_train, y_train)
            self.logger.info("evaluating...")
            train_loss, train_score, train_IC = self.test_epoch(
                x_train, y_train, "train"
            )
            val_loss, val_score, val_IC = self.test_epoch(x_valid, y_valid, "valid")
            test_loss, test_score, test_IC = self.test_epoch(x_test, y_test, "test")
            self.logger.info(
                "train %.6f, valid %.6f, test %.4f"
                % (train_score, val_score, test_score)
            )
            self.logger.info(
                "train IC %.6f, valid IC %.6f, test IC %.4f"
                % (train_IC, val_IC, test_IC)
            )
            evals_result["train"].append(train_score)
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

    def predict(self, dataset: DatasetH, segment="test"):
        if not self.fitted:
            raise ValueError("model is not fitted yet!")

        x_test = dataset.prepare(segment, col_set="feature")
        df_index = x_test.index
        self.model.eval()
        x_values = x_test.values
        preds = []

        if self.use_corr_en:
            (
                daily_index,
                daily_count,
                stocks_daily,
                corr_en_shuffled,
            ) = self.get_daily_inter(
                x_test, corr_en=self.corr_en["test"], shuffle=False
            )
        else:
            daily_index, daily_count, stocks_daily = self.get_daily_inter(
                x_test, shuffle=False
            )

        for i in range(len(daily_index)):
            idx, count, stks = daily_index[i], daily_count[i], stocks_daily[i]
            if self.use_corr_en:
                corr_A = corr_en_shuffled[i]
                A = self.get_mixed_graph(corr_A, self.rel_encoding)
                self.model.set_graph(rel_encoding=A, device=self.device)
            index = [self.stock_name_list.index(stk) for stk in stks]
            batch = slice(idx, idx + count)
            feature = torch.from_numpy(x_values[batch]).float().to(self.device)

            with torch.no_grad():
                pred = self.model(feature.float(), index).detach().cpu().numpy()

            preds.append(pred)

        return pd.Series(np.concatenate(preds), index=df_index)


class GSLGraphModel(HomographModel):
    def __init__(
        self,
        d_feat=6,
        hidden_size=64,
        num_layers=2,
        dropout=0.0,
        base_model="GRU",
        num_graph_layer=2,
        generate_weight=False,
    ):
        super().__init__(
            d_feat=d_feat,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            base_model=base_model,
        )
        self.d_feat = d_feat
        self.gcn_layers = nn.ModuleList()
        for i in range(num_graph_layer):
            self.gcn_layers.append(dgl.nn.pytorch.GraphConv(hidden_size, hidden_size))
        if generate_weight:
            self.graph_left_generator = nn.Linear(hidden_size, 1)
            self.graph_right_generator = nn.Linear(hidden_size, 1)

        self.generate_weight = generate_weight
        # self.fc_out = nn.Linear(hidden_size * 2, 1)
        self.fc_out = nn.Linear(hidden_size, 1)

    def forward(self, x, index, edge_weight=None):
        if not self.g:
            raise ValueError("graph not specified")
        # x: [N, F*T]
        x = x.reshape(len(x), self.d_feat, -1)  # [N, F, T]
        x = x.permute(0, 2, 1)  # [N, T, F]
        out, _ = self.rnn(x)
        h = out[:, -1, :]
        outputs_h = h
        subgraph = dgl.node_subgraph(self.g, index)

        with self.g.local_scope():
            if edge_weight is not None:
                for i, layer in enumerate(self.gcn_layers):
                    outputs_h = layer(subgraph, outputs_h, edge_weight=edge_weight)
                    outputs_h = F.relu(outputs_h)
            else:
                if not self.generate_weight:
                    for i, layer in enumerate(self.gcn_layers):
                        outputs_h = layer(subgraph, outputs_h)
                        outputs_h = F.relu(outputs_h)
                else:
                    # generate edge weight by f(x_1, x_2)
                    left = self.graph_left_generator(outputs_h).reshape(-1, 1)
                    right = self.graph_right_generator(outputs_h).reshape(1, -1)
                    batch_size = x.shape[0]
                    all_one = torch.ones(batch_size, 1, dtype=torch.float32).to(
                        x.device
                    )
                    predicted_A = left @ all_one.T + all_one @ right
                    # 现在要算所有股票 pair wise的相似度, 每个epoch 10min, 过于慢了  还是实现一下基于DGL的边的算法

                    edges = subgraph.edges()
                    generated_edge_weight = torch.zeros(len(edges[0])).to(x.device)
                    for i in range(len(edges[0])):
                        generated_edge_weight[i] = predicted_A[edges[0][i], edges[1][i]]
                    for i, layer in enumerate(self.gcn_layers):
                        outputs_h = layer(
                            subgraph,
                            outputs_h,
                            edge_weight=torch.sigmoid(F.relu(generated_edge_weight)),
                        )
                        outputs_h = F.relu(outputs_h)

        # return self.fc_out(torch.cat([h, outputs_h], dim=1)).squeeze()
        return self.fc_out(outputs_h).squeeze()


class GATModel(HomographModel):
    def __init__(
        self,
        d_feat=6,
        hidden_size=64,
        num_layers=2,
        dropout=0.0,
        base_model="GRU",
        num_graph_layer=2,
        heads=None,
    ):
        super().__init__(
            d_feat=d_feat,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            base_model=base_model,
        )

        self.hidden_size = hidden_size
        self.d_feat = d_feat
        # self.fc_out = nn.Linear(hidden_size * 2, 1)
        self.fc_out = nn.Linear(hidden_size, 1)
        self.leaky_relu = nn.LeakyReLU()
        self.softmax = nn.Softmax(dim=1)

        self.num_graph_layer = num_graph_layer
        self.gat_layers = nn.ModuleList()

        if not heads:  # set default attention heads
            heads = [1] * num_graph_layer
        heads = [1] + heads

        for i in range(num_graph_layer - 1):
            self.gat_layers.append(
                dglnn.GATConv(
                    hidden_size * heads[i],
                    hidden_size,
                    heads[i + 1],
                    feat_drop=0.6,
                    attn_drop=0.6,
                    activation=F.elu,
                )
            )
        self.gat_layers.append(
            dglnn.GATConv(
                hidden_size * heads[-2],
                hidden_size,
                heads[-1],
                feat_drop=0.6,
                attn_drop=0.6,
                activation=None,
            )
        )

    def get_attention(self, x, index):
        if not self.g:
            raise ValueError("graph not specified")

        # x: [N, F*T]
        x = x.reshape(len(x), self.d_feat, -1)  # [N, F, T]
        x = x.permute(0, 2, 1)  # [N, T, F]
        out, _ = self.rnn(x)
        h = out[:, -1, :]

        subgraph = dgl.node_subgraph(self.g, index)
        attn = []
        for i, layer in enumerate(self.gat_layers):
            h, layer_attentionm = layer(subgraph, h, get_attention=True)  # [E,*,H,1]
            attn.append(layer_attentionm)
            if i == self.num_graph_layer - 1:  # last layer
                h = h.mean(1)
            else:  # other layer(s)
                h = h.flatten(1)
        return attn

    def forward(self, x, index):
        if not self.g:
            raise ValueError("graph not specified")
        # x: [N, F*T]
        x = x.reshape(len(x), self.d_feat, -1)  # [N, F, T]
        x = x.permute(0, 2, 1)  # [N, T, F]
        out, _ = self.rnn(x)
        h = out[:, -1, :]
        outputs_h = h
        subgraph = dgl.node_subgraph(self.g, index)
        for i, layer in enumerate(self.gat_layers):
            outputs_h = layer(subgraph, outputs_h)
            if i == self.num_graph_layer - 1:  # last layer
                outputs_h = outputs_h.mean(1)
            else:  # other layer(s)
                outputs_h = outputs_h.flatten(1)
        # return self.fc_out(torch.cat([h, outputs_h], dim=1)).squeeze()
        return self.fc_out(h).squeeze()


class simpleHeteroGATConv(nn.Module):
    def __init__(
        self,
        edge_feats,
        num_etypes,
        in_feats,
        out_feats,
        num_heads,
        feat_drop=0.0,
        attn_drop=0.0,
        negative_slope=0.2,
        residual=False,
        activation=None,
        allow_zero_in_degree=False,
        bias=False,
        alpha=0.0,
    ):
        super(simpleHeteroGATConv, self).__init__()
        self._edge_feats = edge_feats
        self._num_heads = num_heads
        self._in_feats = in_feats
        self._out_feats = out_feats
        self._in_src_feats = self._in_dst_feats = in_feats
        self._allow_zero_in_degree = allow_zero_in_degree
        self.edge_emb = nn.Parameter(torch.FloatTensor(size=(num_etypes, edge_feats)))

        in_dim = None
        for name in in_feats:
            if in_dim:
                assert in_dim == in_feats[name]
            else:
                in_dim = in_feats[name]
        self.fc = nn.Linear(in_dim, out_feats * num_heads, bias=False)
        self.fc_e = nn.Linear(edge_feats, edge_feats * num_heads, bias=False)
        self.attn_l = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))
        self.attn_r = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))
        self.attn_e = nn.Parameter(torch.FloatTensor(size=(1, num_heads, edge_feats)))
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        if residual:
            in_dim = None
            for name in in_feats:
                if in_dim:
                    assert in_dim == in_feats[name]
                else:
                    in_dim = in_feats[name]
            if in_dim != num_heads * out_feats:
                self.res_fc = nn.Linear(in_dim, num_heads * out_feats, bias=False)
            else:
                self.res_fc = Identity()
        else:
            self.register_buffer("res_fc", None)
        self.reset_parameters()
        self.activation = activation
        self.bias = bias
        if bias:
            self.bias_param = nn.Parameter(torch.zeros((1, num_heads, out_feats)))
        self.alpha = alpha

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")

        nn.init.xavier_normal_(self.fc.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_l, gain=gain)
        nn.init.xavier_normal_(self.attn_r, gain=gain)
        nn.init.xavier_normal_(self.attn_e, gain=gain)
        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)
        nn.init.xavier_normal_(self.fc_e.weight, gain=gain)
        nn.init.normal_(self.edge_emb, 0, 1)

    def set_allow_zero_in_degree(self, set_value):
        self._allow_zero_in_degree = set_value

    def forward(self, graph, nfeat, res_attn=None):
        with graph.local_scope():
            funcs = {}

            for ntype in graph.ntypes:
                h = self.feat_drop(nfeat[ntype])
                feat = self.fc(h).view(-1, self._num_heads, self._out_feats)

                graph.nodes[ntype].data["ft"] = feat
                if self.res_fc is not None:
                    graph.nodes[ntype].data["h"] = h

            for src, etype, dst in graph.canonical_etypes:
                feat_src = graph.nodes[src].data["ft"]
                feat_dst = graph.nodes[dst].data["ft"]
                el = (feat_src * self.attn_l).sum(dim=-1).unsqueeze(-1)
                graph.nodes[src].data["el"] = el
                er = (feat_dst * self.attn_r).sum(dim=-1).unsqueeze(-1)
                graph.nodes[dst].data["er"] = er
                e_feat = self.edge_emb[int(etype)].unsqueeze(0)
                e_feat = self.fc_e(e_feat).view(-1, self._num_heads, self._edge_feats)
                ee = (
                    (e_feat * self.attn_e)
                    .sum(dim=-1)
                    .unsqueeze(-1)
                    .expand(graph.number_of_edges(etype), self._num_heads, 1)
                )
                graph.apply_edges(fn.u_add_v("el", "er", "e"), etype=etype)
                graph.edges[etype].data["a"] = self.leaky_relu(
                    graph.edges[etype].data.pop("e") + ee
                )

            hg = dgl.to_homogeneous(graph, edata=["a"])
            a = self.attn_drop(edge_softmax(hg, hg.edata.pop("a")))
            e_t = hg.edata["_TYPE"]

            for src, etype, dst in graph.canonical_etypes:
                t = graph.get_etype_id(etype)
                graph.edges[etype].data["a"] = a[e_t == t]
                if res_attn is not None:
                    graph.edges[etype].data["a"] = (
                        graph.edges[etype].data["a"] * (1 - self.alpha)
                        + res_attn[etype] * self.alpha
                    )
                funcs[etype] = (fn.u_mul_e("ft", "a", "m"), fn.sum("m", "ft"))

            graph.multi_update_all(funcs, "sum")
            rst = graph.ndata.pop("ft")
            graph.edata.pop("el")
            graph.edata.pop("er")
            if self.res_fc is not None:
                for ntype in graph.ntypes:
                    rst[ntype] = (
                        self.res_fc(graph.nodes[ntype].data["h"]).view(
                            graph.nodes[ntype].data["h"].shape[0],
                            self._num_heads,
                            self._out_feats,
                        )
                        + rst[ntype]
                    )

            if self.bias:
                for ntype in graph.ntypes:
                    rst[ntype] = rst[ntype] + self.bias_param

            if self.activation:
                for ntype in graph.ntypes:
                    rst[ntype] = self.activation(rst[ntype])
            res_attn = {e: graph.edges[e].data["a"].detach() for e in graph.etypes}
            graph.edata.pop("a")
            return rst, res_attn


class SimpleHeteroHGN(HeterographModel):
    def __init__(
        self,
        d_feat,
        edge_dim,
        num_etypes,
        num_hidden,
        num_layers,
        dropout,
        feat_drop,
        attn_drop,
        negative_slope,
        residual,
        alpha,
        base_model,
        num_graph_layer,
        heads=None,
    ):
        super(SimpleHeteroHGN, self).__init__(
            base_model=base_model,
            d_feat=d_feat,
            hidden_size=num_hidden,
            num_layers=num_layers,
            dropout=dropout,
        )

        self.num_layers = num_graph_layer
        self.gat_layers = nn.ModuleList()
        self.activation = F.elu

        in_dims = {"none": num_hidden, self.target_type: num_hidden}
        self.gat_layers.append(
            simpleHeteroGATConv(
                edge_dim,
                num_etypes,
                in_dims,
                num_hidden,
                heads[0],
                feat_drop,
                attn_drop,
                negative_slope,
                False,
                self.activation,
                alpha=alpha,
            )
        )
        for l in range(1, self.num_layers):
            in_dims = {n: num_hidden * heads[l - 1] for n in in_dims}
            self.gat_layers.append(
                simpleHeteroGATConv(
                    edge_dim,
                    num_etypes,
                    in_dims,
                    num_hidden,
                    heads[l],
                    feat_drop,
                    attn_drop,
                    negative_slope,
                    residual,
                    self.activation,
                    alpha=alpha,
                )
            )
        in_dims = num_hidden * heads[-2]
        self.fc_out = nn.Linear(in_dims, 1)

    def get_attention(self, x, index):
        # TODO
        pass

    def forward(self, x, index):
        if not self.g:
            raise ValueError("graph not specified")
        # x: [N, F*T]
        x = x.reshape(len(x), self.d_feat, -1)  # [N, F, T]
        x = x.permute(0, 2, 1)  # [N, T, F]
        out, _ = self.rnn(x)

        h = {self.target_type: out[:, -1, :], "none": self.none_feature}
        res_attn = None
        subgraph = dgl.node_subgraph(self.g, index)
        for l in range(self.num_layers):
            h, res_attn = self.gat_layers[l](subgraph, h, res_attn=res_attn)
            h = {n: h[n].flatten(1) for n in h}
        h = h[self.target_ntype]
        return self.fc_out(h).squeeze()
