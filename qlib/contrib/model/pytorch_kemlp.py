from __future__ import division
from __future__ import print_function

import json
import pickle
from collections import defaultdict

import os
import gc
import numpy as np
import pandas as pd
from typing import Callable, Optional, Text, Union, List, Dict
from sklearn.metrics import roc_auc_score, mean_squared_error

import torch
import torch.nn as nn
import torch.optim as optim

from .pytorch_utils import count_parameters
from ...model.base import Model
from ...data.dataset import DatasetH
from ...data.dataset.handler import DataHandlerLP
from ...data.dataset.weight import Reweighter
from ...utils import (
    auto_filter_kwargs,
    init_instance_by_config,
    unpack_archive_with_buffer,
    save_multiple_parts_file,
    get_or_create_path,
)
from ...log import get_module_logger
from ...workflow import R
from qlib.contrib.meta.data_selection.utils import ICLoss
from torch.nn import DataParallel
from qlib.contrib.model.pytorch_nn import DNNModelPytorch, AverageMeter


class KEMLPPytorch(DNNModelPytorch):
    def __init__(self,
                 kg_embedding_path,
                 stock_id_table_path,
                 date_table_path=None,
                 **kwargs):
        super().__init__(**kwargs)
        if stock_id_table_path.endswith('json'):
            with open(stock_id_table_path) as f:
                self.stock_id_table = json.load(f)
        else:
            with open(stock_id_table_path, 'rb') as f:
                self.stock_id_table = pickle.load(f)
        if kg_embedding_path.endswith('csv'):
            self.kg_embedding = pd.read_csv(kg_embedding_path, header=None).values
        elif kg_embedding_path.endswith('npz'):
            self.kg_embedding = np.load(kg_embedding_path)['embeddings']
        elif kg_embedding_path.endswith('npy'):
            self.kg_embedding = np.load(kg_embedding_path)
        elif kg_embedding_path.endswith('pkl'):
            self.kg_embedding = pd.read_pickle(kg_embedding_path).values
        else:
            raise NotImplementedError
        self.kg_embedding = (self.kg_embedding - self.kg_embedding.mean(-2, keepdims=True)) / self.kg_embedding.std(-2, keepdims=True)

        self.static = self.kg_embedding.ndim == 2
        if not self.static:
            self.date_table = pd.read_csv(date_table_path, header=None)

    def fit(
        self,
        dataset: DatasetH,
        evals_result=dict(),
        verbose=True,
        save_path=None,
        reweighter=None,
    ):
        has_valid = "valid" in dataset.segments
        segments = ["train", "valid"]
        vars = ["x", "y", "w"]
        all_df = defaultdict(dict)  # x_train, x_valid y_train, y_valid w_train, w_valid
        all_t = defaultdict(dict)  # tensors
        for seg in segments:
            if seg in dataset.segments:
                # df_train df_valid
                df = dataset.prepare(
                    seg, col_set=["feature", "label"], data_key=self.valid_key if seg == "valid" else DataHandlerLP.DK_L
                )
                df = df[df.index.get_level_values(1).isin(self.stock_id_table.keys())]
                all_df["x"][seg] = df["feature"]
                all_df["y"][seg] = df["label"].copy()  # We have to use copy to remove the reference to release mem
                if reweighter is None:
                    all_df["w"][seg] = pd.DataFrame(np.ones_like(all_df["y"][seg].values), index=df.index)
                elif isinstance(reweighter, Reweighter):
                    all_df["w"][seg] = pd.DataFrame(reweighter.reweight(df))
                else:
                    raise ValueError("Unsupported reweighter type.")

                # get tensors
                for v in vars:
                    all_t[v][seg] = torch.from_numpy(all_df[v][seg].values).float()
                    # if seg == "valid": # accelerate the eval of validation
                    all_t[v][seg] = all_t[v][seg].to(self.device)  # This will consume a lot of memory !!!!

                kg_emb = torch.from_numpy(self.kg_embedding).float()
                kg_emb = kg_emb.to(self.device)
                idx = torch.tensor([self.stock_id_table[_id] for _id in df.index.get_level_values(1)])
                if self.static:
                    kg_emb = kg_emb[idx]
                else:
                    date_idx = df.index.get_level_values(0).map(
                        lambda date: self.date_table[(self.date_table.iloc[:, 0] <= str(date)) &
                                                     (self.date_table.iloc[:, 1] >= str(date))].index[0])
                    date_idx = torch.LongTensor(date_idx.to_numpy()).to(self.device)
                    kg_emb = kg_emb[date_idx, idx]
                all_t['x'][seg] = torch.cat([all_t['x'][seg], kg_emb], -1)

                evals_result[seg] = []
                # free memory
                del df
                del all_df["x"]
                gc.collect()

        save_path = get_or_create_path(save_path)
        stop_steps = 0
        train_loss = 0
        best_loss = np.inf
        # train
        self.logger.info("training...")
        self.fitted = True
        # return
        # prepare training data
        train_num = all_t["y"]["train"].shape[0]

        for step in range(1, self.max_steps + 1):
            if stop_steps >= self.early_stop_rounds:
                if verbose:
                    self.logger.info("\tearly stop")
                break
            loss = AverageMeter()
            self.dnn_model.train()
            self.train_optimizer.zero_grad()
            choice = np.random.choice(train_num, self.batch_size)
            x_batch_auto = all_t["x"]["train"][choice].to(self.device)
            y_batch_auto = all_t["y"]["train"][choice].to(self.device)
            w_batch_auto = all_t["w"]["train"][choice].to(self.device)

            # forward
            preds = self.dnn_model(x_batch_auto)
            cur_loss = self.get_loss(preds, w_batch_auto, y_batch_auto, self.loss_type)
            cur_loss.backward()
            self.train_optimizer.step()
            loss.update(cur_loss.item())
            R.log_metrics(train_loss=loss.avg, step=step)

            # validation
            train_loss += loss.val
            # for evert `eval_steps` steps or at the last steps, we will evaluate the model.
            if step % self.eval_steps == 0 or step == self.max_steps:
                if has_valid:
                    stop_steps += 1
                    train_loss /= self.eval_steps

                    with torch.no_grad():
                        self.dnn_model.eval()

                        # forward
                        preds = self._nn_predict(all_t["x"]["valid"], return_cpu=False)
                        cur_loss_val = self.get_loss(preds, all_t["w"]["valid"], all_t["y"]["valid"], self.loss_type)
                        loss_val = cur_loss_val.item()
                        metric_val = (
                            self.get_metric(
                                preds.reshape(-1), all_t["y"]["valid"].reshape(-1), all_df["y"]["valid"].index
                            )
                            .detach()
                            .cpu()
                            .numpy()
                            .item()
                        )
                        R.log_metrics(val_loss=loss_val, step=step)
                        R.log_metrics(val_metric=metric_val, step=step)

                        if self.eval_train_metric:
                            metric_train = (
                                self.get_metric(
                                    self._nn_predict(all_t["x"]["train"], return_cpu=False),
                                    all_t["y"]["train"].reshape(-1),
                                    all_df["y"]["train"].index,
                                )
                                .detach()
                                .cpu()
                                .numpy()
                                .item()
                            )
                            R.log_metrics(train_metric=metric_train, step=step)
                        else:
                            metric_train = np.nan
                    if verbose:
                        self.logger.info(
                            f"[Step {step}]: train_loss {train_loss:.6f}, valid_loss {loss_val:.6f}, train_metric {metric_train:.6f}, valid_metric {metric_val:.6f}"
                        )
                    evals_result["train"].append(train_loss)
                    evals_result["valid"].append(loss_val)
                    if loss_val < best_loss:
                        if verbose:
                            self.logger.info(
                                "\tvalid loss update from {:.6f} to {:.6f}, save checkpoint.".format(
                                    best_loss, loss_val
                                )
                            )
                        best_loss = loss_val
                        self.best_step = step
                        R.log_metrics(best_step=self.best_step, step=step)
                        stop_steps = 0
                        torch.save(self.dnn_model.state_dict(), save_path)
                    train_loss = 0
                    # update learning rate
                    if self.scheduler is not None:
                        auto_filter_kwargs(self.scheduler.step, warning=False)(metrics=cur_loss_val, epoch=step)
                    R.log_metrics(lr=self.get_lr(), step=step)
                else:
                    # retraining mode
                    if self.scheduler is not None:
                        self.scheduler.step(epoch=step)

        if has_valid:
            # restore the optimal parameters after training
            self.dnn_model.load_state_dict(torch.load(save_path, map_location=self.device))
        if self.use_gpu:
            torch.cuda.empty_cache()

    def predict(self, dataset: DatasetH, segment: Union[Text, slice] = "test"):
        if not self.fitted:
            raise ValueError("model is not fitted yet!")
        x_test_pd = dataset.prepare(segment, col_set="feature", data_key=DataHandlerLP.DK_I)
        # x_test_pd = x_test_pd[x_test_pd.index.get_level_values(1).isin(self.stock_id_table.keys())]

        kg_emb = torch.from_numpy(self.kg_embedding).float()
        # kg_emb = kg_emb.to(self.device)
        idx = torch.tensor([self.stock_id_table[_id] for _id in x_test_pd.index.get_level_values(1)])
        if self.static:
            kg_emb = kg_emb[idx]
        else:
            date_idx = x_test_pd.index.get_level_values(0).map(
                lambda date: self.date_table[(self.date_table.iloc[:, 0] <= str(date)) &
                                             (self.date_table.iloc[:, 1] >= str(date))].index[0])
            date_idx = torch.LongTensor(date_idx.to_numpy())
            kg_emb = kg_emb[date_idx, idx]

        x_test = torch.Tensor(x_test_pd.values)
        x_test = torch.cat([x_test, kg_emb], -1)
        preds = self._nn_predict(x_test)
        return pd.Series(preds.reshape(-1), index=x_test_pd.index)

