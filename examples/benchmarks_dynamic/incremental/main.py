import copy
import os.path
import time
import traceback
from pathlib import Path
from pprint import pprint
from typing import Optional, Dict, Union, List
from torchsummary import summary

import sys
import pandas as pd
import fire
import torch
import yaml
from torch import nn


DIRNAME = Path(__file__).absolute().resolve().parent
sys.path.append(str(DIRNAME))
sys.path.append(str(DIRNAME.parent.parent.parent))


from src.model import IncrementalManager, DoubleAdaptManager
from src import utils


class Benchmark:
    def __init__(self, data_dir="cn_data", market="csi300", model_type="linear", alpha=360,
                 lr=0.001, early_stop=200, horizon=1, rank_label=True,
                 h_path: Optional[str] = None,
                 train_start: Optional[str] = None,
                 test_start: Optional[str] = None,
                 test_end: Optional[str] = None, ) -> None:
        self.data_dir = data_dir
        self.market = market
        self.horizon = horizon
        self.model_type = model_type
        self.h_path = h_path
        self.train_start = train_start
        self.test_start = test_start
        self.test_end = test_end
        self.alpha = alpha
        self.rank_label = rank_label
        self.lr = lr
        self.early_stop = early_stop

    def basic_task(self):
        """For fast training rolling"""
        if self.model_type == "MLP":
            conf_path = (DIRNAME.parent.parent / "benchmarks" / "MLP" / "workflow_config_mlp_Alpha{}.yaml".format(
                self.alpha))
            filename = "MLP_alpha{}_handler_horizon{}.pkl".format(self.alpha, self.horizon)
        else:
            conf_path = (
                        DIRNAME.parent.parent / "benchmarks" / self.model_type / "workflow_config_{}_Alpha{}.yaml".format(
                    self.model_type.lower(), self.alpha))
            filename = "alpha{}_handler_horizon{}.pkl".format(self.alpha, self.horizon)
        filename = f"{self.data_dir}_{self.market}_rank{self.rank_label}_{filename}"
        h_path = DIRNAME.parent / "baseline" / filename
        # h_path = DIRNAME / filename

        if self.h_path is not None:
            h_path = Path(self.h_path)

        with conf_path.open("r") as f:
            conf = yaml.safe_load(f)

        # modify dataset horizon
        conf["task"]["dataset"]["kwargs"]["handler"]["kwargs"]["label"] = [
            "Ref($close, -{}) / Ref($close, -1) - 1".format(self.horizon + 1)
        ]

        if self.market != "csi300":
            conf["task"]["dataset"]["kwargs"]["handler"]["kwargs"]["instruments"] = self.market
            if self.data_dir == "us_data":
                conf["task"]["dataset"]["kwargs"]["handler"]["kwargs"]["label"] = [
                    "Ref($close, -{}) / $close - 1".format(self.horizon)
                ]

        batch_size = 1
        for k, v in {'early_stop': self.early_stop, "batch_size": batch_size, "lr": self.lr, "seed": None, }.items():
            if k in conf["task"]["model"]["kwargs"]:
                conf["task"]["model"]["kwargs"][k] = v
        if conf["task"]["model"]["class"] == "TransformerModel":
            conf["task"]["model"]["kwargs"]["dim_feedforward"] = 32
            conf["task"]["model"]["kwargs"]["reg"] = 0

        task = conf["task"]

        h_conf = task["dataset"]["kwargs"]["handler"]

        if not h_path.exists():
            from qlib.utils import init_instance_by_config
            h = init_instance_by_config(h_conf)
            h.to_pickle(h_path, dump_all=True)
            print('Save handler file to', h_path)

        task["dataset"]["kwargs"]["handler"] = f"file://{h_path}"

        if self.train_start is not None:
            seg = task["dataset"]["kwargs"]["segments"]["train"]
            task["dataset"]["kwargs"]["segments"]["train"] = pd.Timestamp(self.train_start), seg[1]

        if self.test_start is not None:
            seg = task["dataset"]["kwargs"]["segments"]["train"]
            task["dataset"]["kwargs"]["segments"]["test"] = pd.Timestamp(self.test_start), seg[1]

        if self.test_end is not None:
            seg = task["dataset"]["kwargs"]["segments"]["test"]
            task["dataset"]["kwargs"]["segments"]["test"] = seg[0], pd.Timestamp(self.test_end)
        # print(task)
        return task


# where the root_path to data can be found
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
            data_dir="cn_data",
            root_path='./dataset_j/dataset_jj/',
            calendar_path=None,
            market="csi300",
            horizon=1,
            alpha=360,
            x_dim=None,
            step=2,
            model_name="ALSTM",
            lr=0.001,
            lr_ma=None,
            lr_da=0.01,
            lr_x=None,
            lr_y=None,
            online_lr: dict = None,
            reg=0.5,
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

        # !!!!!! !!!!!!
        calendar = pd.read_pickle("dataset_j/calendar.pkl")  # pd.series
        self.ta = utils.TimeAdjuster(calendar)

        self.market = market
        if self.market == "csi500":
            self.benchmark = "SH000905"
        else:
            self.benchmark = "SH000300"
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

        # self.segments = self.basic_task["dataset"]["kwargs"]["segments"]
        if test_start is not None:
            self.segments['test'] = (test_start, self.segments['test'][1])
        if test_end is not None:
            self.segments['test'] = (self.segments['test'][0], test_end)

        self.test_slice = slice(self.ta.align_time(self.segments['test'][0], tp_type='start'),
                                self.ta.align_time(self.segments['test'][1], tp_type='end'))

        self.h_path = h_path
        self.preprocess_tensor = preprocess_tensor
        self.use_extra = use_extra

        # self.factor_num = 6 if self.alpha == 360 else 20
        self.factor_num = 6
        self.x_dim = x_dim if x_dim else (360 if self.alpha == 360 else 20 * 20)
        print('Experiment name:', self.experiment_name)

    @property
    def experiment_name(self):
        return f"{self.market}_{self.model_name}_alpha{self.alpha}_horizon{self.horizon}_step{self.step}" \
               f"_rank{self.rank_label}_{self.tag}"

    @property
    def basic_task(self):
        return Benchmark(
            data_dir=self.data_dir,
            market=self.market,
            model_type=self.model_name,
            horizon=self.horizon,
            rank_label=self.rank_label,
            alpha=self.alpha,
            lr=self.lr,
            early_stop=200,
            h_path=self.h_path,
            test_start=self.test_slice.start,
            test_end=self.test_slice.stop,
        ).basic_task()

    def _load_data(self):
        # FIXME: load your own data!
        """
        Returns:
            pd.DataFrame:
                the index col is pd.MultiIndex with the datetime as level 0 and the stock ID as level 1;
                the col named 'feature' contains the stock feature vectors;
                the col named 'label' contains the ground-truth labels.
        """
        from qlib.utils import init_instance_by_config
        import qlib

        qlib.init(provider_uri=self.provider_uri, region="us" if self.data_dir == "us_data" else "cn",)

        dataset_conf = self.basic_task['dataset']

        print('dataset conf is', dataset_conf)

        return init_instance_by_config(dataset_conf).handler._learn

    def _init_model(self) -> nn.Module:
        # FIXME: init your own model!
        from qlib.utils import init_instance_by_config

        if self.basic_task["model"]["class"] == "LinearModel":
            return nn.Linear(self.x_dim, 1, bias=False)
        else:
            model = init_instance_by_config(self.basic_task["model"])
            for child in model.__dict__.values():
                if isinstance(child, nn.Module):
                    return child

    def offline_training(self, segments: Dict[str, tuple] = None, data: pd.DataFrame = None, reload_path=None, save_path=None):
        model = self._init_model()

        if self.naive:
            framework = IncrementalManager(model, x_dim=self.x_dim, lr_model=self.lr, begin_valid_epoch=0)
        else:
            framework = DoubleAdaptManager(model, x_dim=self.x_dim, lr_model=self.lr, weight_decay=self.weight_decay,
                                           first_order=self.first_order, begin_valid_epoch=0, factor_num=self.factor_num,
                                           lr_da=self.lr_da, lr_ma=self.lr_ma, online_lr=self.online_lr,
                                           lr_x=self.lr_x, lr_y=self.lr_y,
                                           adapt_x=self.adapt_x, adapt_y=self.adapt_y, reg=self.reg,
                                           num_head=self.num_head, temperature=self.temperature)
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

            model = self._init_model()
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

    def _evaluate_metrics(self, pred: pd.DataFrame):
        from qlib.utils import init_instance_by_config
        from qlib.data.dataset import DataHandlerLP

        """        
        Note that the labels in pred_y_all are preprocessed. IC should be calculated by the raw labels. 
        """
        ds = init_instance_by_config(self.basic_task["dataset"])
        label_all = ds.prepare(segments="test", col_set="label", data_key=DataHandlerLP.DK_R)
        label_all = label_all.dropna(axis=0)
        df = pred.loc[label_all.index]
        df['label'] = label_all.values

        ic = df.groupby('datetime').apply(lambda df: df["pred"].corr(df["label"]))
        ric = df.groupby('datetime').apply(lambda df: df["pred"].corr(df["label"], method="spearman"))
        metrics = {
            "IC": ic.mean(),
            "ICIR": ic.mean() / ic.std(),
            "Rank IC": ric.mean(),
            "Rank ICIR": ric.mean() / ric.std(),
        }
        pprint(metrics)

    def workflow(self, checkpoint_dir: str = "./checkpoints/", reload_path: str = None):
        if checkpoint_dir:
            if not os.path.exists(checkpoint_dir):
                os.mkdir(checkpoint_dir)
            save_path = os.path.join(checkpoint_dir, f"{self.experiment_name}.pt")
        else:
            save_path = None

        data = pd.read_pickle("dataset_j/my_data.pkl")

        # print(self.segments)
        assert data.index[0][0] <= self.ta.align_time(self.segments['train'][0], tp_type='start')
        assert data.index[-1][0] >= self.ta.align_time(self.segments['test'][-1], tp_type='end')

        framework = self.offline_training(data=data, save_path=save_path, reload_path=reload_path)
        pred_y_all = self.online_training(data=data, framework=framework)

        torch.save(pred_y_all, "outputs/pre_y.pkl")

        self._evaluate_metrics(pred_y_all)


if __name__ == "__main__":
    print(sys.argv)
    # fire.Fire(IncrementalExp)
    m = IncrementalExp()
    m.workflow()
