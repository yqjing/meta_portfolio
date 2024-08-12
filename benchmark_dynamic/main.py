import copy
import os.path
import time
import traceback
from pathlib import Path
from pprint import pprint
from typing import Optional, Dict, Union, List

import sys
import pandas as pd
import torch
import yaml
from torch import nn


DIRNAME = Path(__file__).absolute().resolve().parent
sys.path.append(str(DIRNAME))


from src.model import IncrementalManager, DoubleAdaptManager
from src import utils
from net import ALSTMModel


class IncrementalExp:
    """
    Example:
        .. code-block:: python

            python -u main.py workflow --model_name GRU --market csi300 --data_dir crowd_data --rank_label False
            --first_order True --adapt_x True --adapt_y True --num_head 8 --tau 10
            --lr 0.001 --lr_da 0.01 --online_lr "{'lr': 0.001, 'lr_da': 0.001, 'lr_ma': 0.001}"
    """

    def __init__(
            self,
            data_name, # string
            data_dir="cn_data",
            root_path='./dataset_j',
            calendar_path=None,
            market="Joint_Portfolio",
            horizon= 5 * 2,
            alpha= 360,
            x_dim= 11 * 60,
            step=10,  # 2
            model_name="ALSTM",
            lr=0.001,
            lr_ma=None,
            lr_da=0.01,
            lr_x=None,
            lr_y=None,
            online_lr: dict = None,
            reg=1, # 0.5
            weight_decay=0,
            num_head=12,   # 8
            tau=10,
            first_order=True,
            adapt_x=True,
            adapt_y=True,
            naive=False,
            preprocess_tensor=True,
            use_extra=False,
            tag=None,
            rank_label=False,
            h_path=None,
            test_start=None,
            test_end=None,
    ):
        """
        Args:
            data_dir (str):
                source data dictionary under root_path
            root_path (str):
                the root path of source data. '~/' by default.
            calendar_path (str):
                the path of calendar. If None, use '~/cn_data/calendar/days.txt'.
            market (str):
                'csi300' or 'csi500'
            horizon (int):
                define the stock price trend
            alpha (int):
                360 or 158
            x_dim (int):
                the dimension of stock features (e.g., factor_num * time_series_length)
            step (int):
                incremental task interval, i.e., timespan of incremental data or test data
            model_name (str):
                consistent with directory name under examples/benchmarks
            lr (float):
                learning rate of forecast model
            lr_ma (float):
                learning rate of model adapter. If None, use lr.
            lr_da (float):
                learning rate of data adapter
            lr_x (float):
                if both lr_x and lr_y are not None, specify the learning rate of the feature adaptation layer.
            lr_y (float):
                if both lr_x and lr_y are not None, specify the learning rate of the label adaptation layer.
            online_lr (dict):
                learning rates during meta-valid and meta-test. Example: --online lr "{'lr_da': 0, 'lr': 0.0001}".
            reg (float):
                regularization strength
            weight_decay (float):
                L2 regularization of the (Adam) optimizer
            num_head (int):
                number of transformation heads
            tau (float):
                softmax temperature
            first_order (bool):
                whether use first-order approximation version of MAML
            adapt_x (bool):
                whether adapt features
            adapt_y (bool):
                whether adapt labels
            naive (bool):
                if True, degrade to naive incremental baseline; if False, use DoubleAdapt
            preprocess_tensor (bool):
                if False, temporally transform each batch from `numpy.ndarray` to `torch.Tensor` (slow, not recommended)
            use_extra (bool):
                if True, use extra segments for upper-level optimization (not recommended when step is large enough)
            tag (str):
                to distinguish experiment id
            h_path (str):
                prefetched handler file path to load
            test_start (str):
                override the start date of test data
            test_end (str):
                override the end date of test data
        """
        self.data_dir = data_dir
        self.provider_uri = os.path.join(root_path, data_dir)

        self.data_name = data_name
        calendar = pd.read_pickle(f"dataset_j/{data_name}_cal.pkl")  # pd.series
        self.ta = utils.TimeAdjuster(calendar)

        self.market = market
        if self.market == "csi500":
            self.benchmark = "Benchmark1"
        else:
            self.benchmark = "Benchmark2"
        self.step = step
        self.horizon = horizon
        self.model_name = model_name  # downstream forecasting models' type
        self.alpha = alpha
        self.tag = tag
        if self.tag is None:
            self.tag = time.strftime('%Y-%m-%d_%H:%M:%S', time.localtime())
        self.rank_label = rank_label
        self.lr = lr
        self.lr_da = lr_da
        self.lr_ma = lr if lr_ma is None else lr_ma
        self.lr_x = lr_x
        self.lr_y = lr_y
        if online_lr is not None and 'lr' in online_lr:
            online_lr['lr_model'] = online_lr['lr']
        self.online_lr = online_lr
        self.num_head = num_head
        self.temperature = tau
        self.first_order = first_order
        self.naive = False
        self.adapt_x = adapt_x
        self.adapt_y = adapt_y
        self.reg = reg
        self.weight_decay = weight_decay
        self.not_sequence = self.model_name in ["MLP", 'Linear'] and self.alpha == 158

        self.segments = {"train" : ("1991-01-01", "2011-12-31"),
                         "valid" : ("2012-01-01", "2012-12-31"),
                         "test"  : ("2013-01-01", "2017-12-31")}

        if test_start is not None:
            self.segments['test'] = (test_start, self.segments['test'][1])
        if test_end is not None:
            self.segments['test'] = (self.segments['test'][0], test_end)

        self.test_slice = slice(self.ta.align_time(self.segments['test'][0], tp_type='start'),
                                self.ta.align_time(self.segments['test'][1], tp_type='end'))

        self.h_path = h_path
        self.preprocess_tensor = preprocess_tensor
        self.use_extra = use_extra

        self.factor_num = 11
        self.x_dim = x_dim if x_dim else (360 if self.alpha == 360 else 20 * 20)
        print('Jing Sole Experiment name:', self.experiment_name)


    @property
    def experiment_name(self):
        return f"{self.model_name}_factor{self.factor_num}_horizon{self.horizon}_step{self.step}" \
               f"_stopThreshold{0.005}_{self.tag}"


    def offline_training(self, segments: Dict[str, tuple] = None, data: pd.DataFrame = None, reload_path=None, save_path=None):
        # model = self._init_model()
        model = ALSTMModel(d_feat=self.factor_num)

        if self.naive:
            framework = IncrementalManager(model, x_dim=self.x_dim, lr_model=self.lr, begin_valid_epoch=0)
        else:
            framework = DoubleAdaptManager(model, x_dim=self.x_dim, lr_model=self.lr, weight_decay=self.weight_decay,
                                           first_order=self.first_order, begin_valid_epoch=0, factor_num=self.factor_num,
                                           lr_da=self.lr_da, lr_ma=self.lr_ma, online_lr=self.online_lr,
                                           lr_x=self.lr_x, lr_y=self.lr_y,
                                           adapt_x=self.adapt_x, adapt_y=self.adapt_y, reg=self.reg,
                                           num_head=self.num_head, temperature=self.temperature)
        # skip the offline_training by directly loading the state_dict()
        if reload_path is not None:
            framework.load_state_dict(torch.load(reload_path))
            print('Reload experiment', reload_path)
        else:
            if segments is None:
                segments = self.segments
            rolling_tasks = utils.organize_all_tasks(segments, self.ta, step=self.step, trunc_days=self.horizon + 1,
                                                     rtype=utils.TimeAdjuster.SHIFT_SD, use_extra=self.use_extra)

            rolling_tasks_data = {k: utils.get_rolling_data(rolling_tasks[k],
                                                            data=data,
                                                            factor_num=self.factor_num, horizon=self.horizon,
                                                            not_sequence=self.not_sequence,
                                                            sequence_last_dim=self.alpha == 360,
                                                            to_tensor=self.preprocess_tensor)
                                  for k in ['train', 'valid']}
            framework.fit(rolling_tasks_data['train'], rolling_tasks_data['valid'], checkpoint_path=save_path)

        return framework

    def online_training(self, segments: Dict[str, tuple] = None, data: pd.DataFrame = None, reload_path: str = None, framework=None, ):
        """
        Perform incremental learning on the test data.

        Args:
            segments (Dict[str, tuple]):
                The date range of training data, validation data, and test data.
                Example::
                    {
                        'train': ('2008-01-01', '2014-12-31'),
                        'valid': ('2015-01-01', '2016-12-31'),
                        'test': ('2017-01-01', '2020-08-01')
                    }
            data (pd.DataFrame):
                the index col is pd.MultiIndex with the datetime as level 0 and the stock ID as level 1;
                the col named 'feature' contains the stock feature vectors;
                the col named 'label' contains the ground-truth labels.
            reload_path (str):
                if not None, reload checkpoints

        Returns:
            pd.DataFrame:
                the index col is pd.MultiIndex with the datetime as level 0 and the stock ID as level 1;
                the col named 'pred' contains the predictions of the model;
                the col named 'label' contains the ground-truth labels which have been preprocessed and may not be the raw.
        """
        if framework is None:
            assert reload_path is not None

            model = ALSTMModel(d_feat=self.factor_num)
            if self.naive:
                framework = IncrementalManager(model, x_dim=self.x_dim, lr_model=self.lr,
                                               online_lr=self.online_lr, weight_decay=self.weight_decay,
                                               first_order=True, alpha=self.alpha, begin_valid_epoch=0)
            else:
                framework = DoubleAdaptManager(model, x_dim=self.x_dim, lr_model=self.lr, weight_decay=self.weight_decay,
                                               first_order=self.first_order, begin_valid_epoch=0, factor_num=self.factor_num,
                                               lr_da=self.lr_da, lr_ma=self.lr_ma, online_lr=self.online_lr,
                                               lr_x=self.lr_x, lr_y=self.lr_y,
                                               adapt_x=self.adapt_x, adapt_y=self.adapt_y, reg=self.reg,
                                               num_head=self.num_head, temperature=self.temperature)
            # framework.framework.to(framework.framework.device)
            framework.load_state_dict(torch.load(reload_path))
            print('Reload experiment', reload_path)

        if segments is None:
            segments = self.segments
        rolling_tasks = utils.organize_all_tasks(segments, self.ta, step=self.step, trunc_days=self.horizon + 1,
                                                 rtype=utils.TimeAdjuster.SHIFT_SD, use_extra=self.use_extra)
        rolling_tasks_data = utils.get_rolling_data(rolling_tasks['test'],
                                                    data=data,
                                                    factor_num=self.factor_num, horizon=self.horizon,
                                                    not_sequence=self.not_sequence,
                                                    sequence_last_dim=self.alpha == 360,
                                                    to_tensor=self.preprocess_tensor)
        return framework.inference(meta_tasks_test=rolling_tasks_data, date_slice=self.test_slice)

    def _evaluate_metrics(self, data, pred_y_all):
        label = data["label"].iloc[:, 0]
        label.index = label.index.droplevel(1)
        label = label.to_frame()
        label.columns = ["label"]

        pred = pred_y_all['pred']
        pred.index = pred.index.droplevel(1)
        pred = pred.to_frame()
        pred.columns = ["pred"]

        df_pred = pd.merge(pred, label, on = "datetime")
        mse = ((df_pred['pred'].to_numpy() - df_pred['label'].to_numpy()) ** 2).mean(axis=0)
        df_pred.to_pickle("results/df_pred.pkl")

        pprint(df_pred[:20])
        pprint(f"##### test mse is {mse}")

    def workflow(self, checkpoint_dir: str = "./checkpoints/", reload_path: str = None):
        if checkpoint_dir:
            if not os.path.exists(checkpoint_dir):
                os.mkdir(checkpoint_dir)
            save_path = os.path.join(checkpoint_dir, f"{self.experiment_name}.pt")
        else:
            save_path = None

        data = pd.read_pickle(f"dataset_j/{self.data_name}.pkl")

        # print(self.segments)
        assert data.index[0][0] <= self.ta.align_time(self.segments['train'][0], tp_type='start')
        assert data.index[-1][0] >= self.ta.align_time(self.segments['test'][-1], tp_type='end')

        framework = self.offline_training(data=data, save_path=save_path, reload_path=reload_path)
        pred_y_all = self.online_training(data=data, framework=framework)

        self._evaluate_metrics(data, pred_y_all)


if __name__ == "__main__":
    start_time = time.time()
    m = IncrementalExp(data_name="BMI25")
    # if skip_training, set it up here!!! Add the reload_path.
    m.workflow(reload_path=None)
    end_time = time.time()
    duration = end_time - start_time
    print(f"The duration of the program is {duration / 60} minutes.")
