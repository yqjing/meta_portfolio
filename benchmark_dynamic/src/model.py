import sys
import os
# Add the directory containing higher_optim.py to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))


import copy
import os
import typing
from collections import defaultdict, OrderedDict
from typing import Dict, List, Union, Optional, Tuple

import numpy as np

from tqdm import tqdm
import pandas as pd
import torch
from torch import optim, nn
import torch.nn.functional as F
import higher

import higher_optim
from utils import override_state, has_rnn
from net import DoubleAdapt, ForecastModel
from gpu import get_gpu_status

from . import higher_optim
from . import utils
from . import net

# from utils import override_state, has_rnn
# from net import DoubleAdapt, ForecastModel


class IncrementalManager:
    """
    Naive incremental learning framework

    Args:
        model (torch.nn.Module): the stock trend forecasting model
        lr_model (float): the learning rate of the model
        online_lr (dict): the learning rates during online training
        x_dim (int): the total number of stock indicators.
        need_permute (bool): whether to permute the last two dimensions of time-series input,
            specially for Alpha360 with the last dimension as time steps.
        over_patience (int): the patience for early stop.
        begin_valid_epoch (int): which epoch to begin validation. Set a moderate one to reduce training time.
    """
    def __init__(
        self,
        model: nn.Module,
        lr_model: float = 0.001,
        online_lr: dict = None,
        x_dim: int = None,
        need_permute: bool = True,
        over_patience: int = 8000,    # 8
        begin_valid_epoch: int = 0,
        **kwargs
    ):
        self.fitted = False
        self.lr_model = lr_model
        self.online_lr = online_lr
        self.over_patience = over_patience
        self.begin_valid_epoch = begin_valid_epoch
        self.framework = self._init_framework(model, x_dim, lr_model, need_permute=need_permute, **kwargs)
        self.opt = self._init_meta_optimizer(**kwargs)

    def _init_framework(self, model: nn.Module, x_dim: int = None, lr_model=0.001,
                        weight_decay=0.0, need_permute=False, **kwargs):
        return ForecastModel(model, x_dim=x_dim, lr=lr_model, need_permute=need_permute, weight_decay=weight_decay)

    def _init_meta_optimizer(self, **kwargs):
        return self.framework.opt

    def state_dict(self, destination: typing.OrderedDict[str, torch.Tensor]=None, prefix='', keep_vars=False):
        r"""Returns a dictionary containing a whole state of the module and the state of the optimizer.

        Returns:
            dict:
                a dictionary containing a whole state of the module and the state of the optimizer.
        """
        if destination is None:
            destination = OrderedDict()
            destination._metadata = OrderedDict()
        destination['framework'] = self.framework.state_dict()
        destination['framework_opt'] = self.framework.opt.state_dict()
        destination['opt'] = self.opt.state_dict()
        return destination

    def load_state_dict(self, state_dict: typing.OrderedDict[str, torch.Tensor],):
        r"""Copies parameters and buffers from :attr:`state_dict` into
        this module and the optimizer.

        Args:
            dict:
                a dict containing parameters and persistent buffers.
        """
        self.framework.load_state_dict(state_dict['framework'])
        self.framework.opt.load_state_dict(state_dict['framework_opt'])
        self.opt.load_state_dict(state_dict['opt'])

    def override_online_lr_(self):
        if self.online_lr is not None:
            if 'lr_model' in self.online_lr:
                self.lr_model = self.online_lr['lr_model']
                self.opt.param_groups[0]['lr'] = self.online_lr['lr_model']

    def fit(self,
            meta_tasks_train: List[Dict[str, Union[pd.Index, torch.Tensor, np.ndarray]]],
            meta_tasks_val: List[Dict[str, Union[pd.Index, torch.Tensor, np.ndarray]]],
            checkpoint_path=""):
        """Make the trainable parameters of the incremental learning framework fit on the training set.

        Args:
            meta_tasks_train (list): a sequence of incremental learning tasks for
                training the forecasting models (in Naive Incremenal Learning)
                or training the meta-learners (in DoubleAdapt)
            meta_tasks_val (list): a sequence of incremental learning tasks for validation
            checkpoint_path (str): the path of checkpoints
        """
        self.cnt = 0
        self.framework.train()
        torch.set_grad_enabled(True)

        mse_all = []
        best_ic, patience = -1e3, self.over_patience
        best_checkpoint = copy.deepcopy(self.framework.state_dict())
        # for epoch in tqdm(range(100), desc="epoch"):
        
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        epoch_num = 200
        for epoch in range(epoch_num):
            print(f"current epoch: {epoch}")
            print(f"{get_gpu_status(self.framework.device)}")
            
            for phase, task_list in zip(['train', 'val'], [meta_tasks_train, meta_tasks_val]):

                pred_y, ic, mse = self._run_epoch(phase, task_list)

                if phase == "val":
                    print("current validation mse is: ", mse)
                    print("predicted y_val head is: \n", pred_y.head(20))
                    mse_all += [mse]
                    best_checkpoint = copy.deepcopy(self.framework.state_dict())
                    torch.save(mse_all, "results/train_mse.pkl")    # requires torch load to unlock

                    if mse < 0.005 and epoch > 5:
                        patience = -1
                    else:
                        patience = self.over_patience
                        
            if patience <= 0:
                print("mse condition is met: stop training.")
                break
        self.framework.load_state_dict(best_checkpoint)
        self._run_epoch('train', meta_tasks_val)
        self.fitted = True
        if True:
            print('Save checkpoint in the Path:', checkpoint_path)
            torch.save(self.state_dict(), checkpoint_path)

    def _run_epoch(self, phase: str, task_list: List[Dict[str, Union[pd.Index, torch.Tensor, np.ndarray]]],
                   tqdm_show: bool=False):
        # # ======================================
        # # for quick prediction without training
        # if phase == "online":
        #     book_quick = torch.load(f"books/book_{100}.pt")
        #     self.framework.load_state_dict(book_quick['framework'])
        #     self.framework.opt.load_state_dict(book_quick['framework_opt'])
        #     self.opt.load_state_dict(book_quick['opt'])
        # # =======================================

        pred_y_all = []
        indices = np.arange(len(task_list))

        if phase == 'train':
            np.random.shuffle(indices)
        else:
            if phase == "val":
                checkpoint = copy.deepcopy(self.state_dict())
            lr_model = self.lr_model
            self.override_online_lr_()

        self.phase = phase
        counter = 0
        for i in tqdm(indices, desc=phase, mininterval=0.5) if True else indices:
            meta_input = task_list[i]
            if not isinstance(meta_input['X_train'], torch.Tensor):
                meta_input = {
                    k: torch.tensor(v, device=self.framework.device, dtype=torch.float32) if 'idx' not in k else v
                    for k, v in meta_input.items()
                }
            if counter >= 1:
                pred = self._run_task(meta_input, phase, counter, backward = True)
            else:
                pred = self._run_task(meta_input, phase, counter, backward = False)
            counter += 1
            if phase != "train":
                test_idx = meta_input["test_idx"]
                pred_y_all.append(
                    pd.DataFrame(
                        {
                            "pred": pd.Series(pred, index=test_idx),
                            "label": pd.Series(meta_input["y_test"], index=test_idx),
                        }
                    )
                )
        if phase != "train":
            pred_y_all = pd.concat(pred_y_all)
        if phase == "val":
            self.lr_model = lr_model
            self.load_state_dict(checkpoint)
            ic = 21
            # ic = pred_y_all.groupby("datetime").apply(lambda df: df["pred"].corr(df["label"], method="pearson")).mean()
            # print(ic)
            # compute the mse loss
            mse = ((pred_y_all['pred'].to_numpy() - pred_y_all['label'].to_numpy()) ** 2).mean(axis=0)
            return pred_y_all, ic, mse
        return pred_y_all, None, None

    def _run_task(self, meta_input: Dict[str, Union[pd.Index, torch.Tensor]], phase: str, counter, backward):
        """ A single naive incremental learning task """
        self.framework.opt.zero_grad()
        # only one forecaster being trained in the offline training and updated in the online training 
        y_hat = self.framework(meta_input["X_train"].to(self.framework.device), None)
        loss = self.framework.criterion(y_hat, meta_input["y_train"].to(self.framework.device))
        loss.backward()
        self.framework.opt.step()
        self.framework.opt.zero_grad()
        with torch.no_grad():
            pred = self.framework(meta_input["X_test"].to(self.framework.device), None)
        return pred.detach().cpu().numpy()

    def inference(self, meta_tasks_test: List[Dict[str, Union[pd.DataFrame, np.ndarray, torch.Tensor]]],
                  date_slice: slice = slice(None, None)):
        """
        Perform incremental learning on the test set.

        Args:
            meta_tasks_test (List[Dict[str, Union[pd.DataFrame, np.ndarray, torch.Tensor]]])
            test_start (Optional[pd.Timestamp]), test_end (Optional[pd.Timestamp]):
                Only return predictions and labels during this range

        Returns:
            pd.DataFrame:
                the index col is pd.MultiIndex with the datetime as level 0 and the stock ID as level 1;
                the col named 'pred' contains the predictions of the model;
                the col named 'label' contains the ground-truth labels which have been preprocessed and may not be the raw.
        """
        self.framework.train()
        self.framework.to(self.framework.device)
        pred_y_all, _, _  = self._run_epoch("online", meta_tasks_test, tqdm_show=True)
        pred_y_all = pred_y_all.loc[date_slice]
        return pred_y_all


class DoubleAdaptManager(IncrementalManager):
    r"""
    A meta-learning based incremental learning framework

    Args:
        model (torch.nn.Module): the stock trend forecasting model
        lr_model (float): the learning rate of the model
        lr_da (float): the learning rate of the data adapter
        lr_ma (float): the learning rate of the model adapter
        reg (float): regularization strength
        adapt_x (bool): whether to perform feature adaptation
        adapt_y (bool): whether to perform label adaptation
        first_order (bool): whether to adopt first-order approximation of MAML
        factor_num (int): the number of indicators at each time step of time-series inputs.
                    Otherwise, the same as :attr:`factor_num`
        x_dim (int): the total number of stock indicators
        need_permute (bool): whether to permute the last two dimensions of time-series input, specially for Alpha360.
        num_head (int): number of adaptation heads
        temperature (float): softmax temperature
        begin_valid_epoch (int): which epoch to begin validation. Set a moderate one to reduce training time.
    """
    def __init__(
        self,
        model: nn.Module,
        lr_model: float = 0.001,
        lr_da: float = 0.01,
        lr_ma: float = 0.001,
        lr_x: float = None,
        lr_y: float = None,
        online_lr: dict = None,
        weight_decay: float = 0,
        reg: float = 0.5,
        adapt_x: bool = True,
        adapt_y: bool = True,
        first_order: bool = True,
        factor_num: int = 6,
        x_dim: int = 360,
        need_permute: bool = True,
        num_head: int = 8,
        temperature: float = 10,
        begin_valid_epoch: int = 0,
    ):
        super(DoubleAdaptManager, self).__init__(model, x_dim=x_dim, lr_model=lr_model, lr_ma=lr_ma, lr_da=lr_da,
                                                 lr_x=lr_x, lr_y=lr_y, online_lr=online_lr, weight_decay=weight_decay,
                                                 need_permute=need_permute,
                                                 factor_num=factor_num, temperature=temperature, num_head=num_head,
                                                 begin_valid_epoch=begin_valid_epoch)
        self.adapt_x = adapt_x
        self.adapt_y = adapt_y
        self.reg = reg
        self.sigma = 1 ** 2 * 2
        self.factor_num = factor_num
        self.num_head = num_head
        self.temperature = temperature
        self.begin_valid_epoch = begin_valid_epoch
        self.first_order = first_order
        # if the forecast model is an RNN and first_order is False, we will disable CUDNN.
        self.has_rnn = has_rnn(self.framework)

    def _init_framework(self, model: nn.Module, x_dim=None, lr_model=0.001, need_permute=False,
                        num_head=8, temperature=10, factor_num=6, lr_ma=None, weight_decay=0, **kwargs):
        return DoubleAdapt(
            model, x_dim=x_dim, lr=lr_model if lr_ma is None else lr_ma, need_permute=need_permute,
            factor_num=factor_num, num_head=num_head, temperature=temperature, weight_decay=weight_decay
        )

    def _init_meta_optimizer(self, lr_da=0.01, lr_x=None, lr_y=None, **kwargs):
        """ NOTE: the optimizer of the model adapter is self.framework.opt """
        if lr_x is None or lr_y is None:
            return optim.Adam(self.framework.meta_params, lr=lr_da)
        else:
            return optim.Adam([{'params': self.framework.teacher_x.parameters(), 'lr': lr_x},
                               {'params': self.framework.teacher_y.parameters(), 'lr': lr_y},])

    def override_online_lr_(self):
        if self.online_lr is not None:
            if 'lr_model' in self.online_lr:
                self.lr_model = self.online_lr['lr_model']
            if 'lr_ma' in self.online_lr:
                self.framework.opt.param_groups[0]['lr'] = self.online_lr['lr_ma']
            if 'lr_da' in self.online_lr:
                self.opt.param_groups[0]['lr'] = self.online_lr['lr_da']
            else:
                if 'lr_x' in self.online_lr:
                    self.opt.param_groups[0]['lr'] = self.online_lr['lr_x']
                if 'lr_y' in self.online_lr:
                    self.opt.param_groups[1]['lr'] = self.online_lr['lr_y']


    def _run_task(self, meta_input: Dict[str, Union[pd.Index, torch.Tensor]], phase: str, counter, backward):

        self.framework.opt.zero_grad()
        self.opt.zero_grad()

        """ Incremental data adaptation & Model adaptation """
        X = meta_input["X_train"].to(self.framework.device)
        # view the shape of X
        # print("The shape of X is", X.shape)

        with higher.innerloop_ctx(
            self.framework.model,
            self.framework.opt,
            copy_initial_weights=False,
            track_higher_grads=not self.first_order,
            override={'lr': [self.lr_model]}
        ) as (fmodel, diffopt):
            with torch.backends.cudnn.flags(enabled=self.first_order or not self.has_rnn):
                y_hat, _ = self.framework(X, model=fmodel, transform=self.adapt_x)
        y = meta_input["y_train"].to(self.framework.device)
        if self.adapt_y:
            raw_y = y
            y = self.framework.teacher_y(X, raw_y, inverse=False)
        loss2 = self.framework.criterion(y_hat, y)
        diffopt.step(loss2)

        """ Online inference """
        if phase != "train" and "X_extra" in meta_input and meta_input["X_extra"].shape[0] > 0:
            X_test = torch.cat([meta_input["X_extra"].to(self.framework.device), meta_input["X_test"].to(self.framework.device), ], 0, )
            y_test = torch.cat([meta_input["y_extra"].to(self.framework.device), meta_input["y_test"].to(self.framework.device), ], 0, )
        else:
            X_test = meta_input["X_test"].to(self.framework.device)
            y_test = meta_input["y_test"].to(self.framework.device)
        pred, X_test_adapted = self.framework(X_test, model=fmodel, transform=self.adapt_x)
        if self.adapt_y:
            pred = self.framework.teacher_y(X_test, pred, inverse=True)
        mask_y = None
        if phase != "train":
            test_begin = len(meta_input["y_extra"]) if "y_extra" in meta_input else 0
            # meta_end = test_begin + meta_input["meta_end"]
            meta_end = 10
            output = pred[test_begin:].detach().cpu().numpy()
            X_test = X_test[:meta_end]
            X_test_adapted = X_test_adapted[:meta_end]
            pred = pred[:meta_end]
            y_test = y_test[:meta_end]
            mask_y = meta_input.get("mask_y")
            if mask_y is not None:
                pred = pred[mask_y[:meta_end]]
                y_test = y_test[mask_y[:meta_end]]
        else:
            output = pred.detach().cpu().numpy()

        """ Optimization of meta-learners """
        # pred, y_test, y_train, y_train_raw, X_train, X_test
        loss = self.framework.criterion(pred, y_test)

        if self.adapt_y:
            if not self.first_order:
                y = self.framework.teacher_y(X, raw_y, inverse=False)
            loss_y = F.mse_loss(y, raw_y) 
            
            if self.first_order:
                with torch.no_grad():
                    pred2, _ = self.framework(X_test_adapted, model=None, transform=False, )
                    pred2 = self.framework.teacher_y(X_test, pred2, inverse=True).detach()
                    if mask_y is not None:
                        pred2 = pred2[mask_y[:meta_end]]
                    loss_old = self.framework.criterion(pred2.view_as(y_test), y_test)
                loss_y = (loss_old.item() - loss.item()) / self.sigma * loss_y + loss_y * self.reg
            else:
                loss_y = loss_y * self.reg

            if counter == 0:
                book = {}
                book['framework'] = self.framework.state_dict()
                book['framework_opt'] = self.framework.opt.state_dict()
                book['opt'] = self.opt.state_dict()
                torch.save(book, f"books/book_{-1}.pt")

            loss_y.backward()
        loss.backward()

        if self.adapt_x or self.adapt_y:
            self.opt.step()
        self.framework.opt.step()

        # torch load lagged state dict
        book_lag = torch.load(f"books/book_{-1}.pt")

        # torch save the current state dict
        book = {}
        book['framework'] = self.framework.state_dict()
        book['framework_opt'] = self.framework.opt.state_dict()
        book['opt'] = self.opt.state_dict()
        torch.save(book, f"books/book_{-1}.pt")

        # load lagged parameters to models
        self.framework.load_state_dict(book_lag['framework'])
        self.framework.opt.load_state_dict(book_lag['framework_opt'])
        self.opt.load_state_dict(book_lag['opt'])

        return output


