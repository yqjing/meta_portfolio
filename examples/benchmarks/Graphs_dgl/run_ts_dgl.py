
import sys
from pathlib import Path
DIRNAME = Path(__file__).absolute().resolve().parent

sys.path.append(str(DIRNAME))
sys.path.append(str(DIRNAME.parent.parent.parent))

import pickle
import argparse
import numpy as np
import os
import torch
import qlib
from qlib.constant import REG_CN, REG_US
from qlib.utils import init_instance_by_config, flatten_dict
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord, PortAnaRecord, SigAnaRecord
from qlib.tests.data import GetData
from qlib.tests.config import CSI300_BENCH, CSI300_DATASET_CONFIG
from qlib.contrib.graph import stock_concept_data, stock_stock_data
from qlib.data import D
from qlib.contrib.interpreter.attentionx import AttentionX
from qlib.contrib.interpreter.xpath import xPath
from qlib.contrib.interpreter.subgraphx import SubgraphXExplainer


def load_graph(market, instrument, start_date, end_date, relation_source='industry', handler_data=None):
    if handler_data is None:
        universe = D.list_instruments(
            D.instruments(instrument), start_time=start_date, end_time=end_date
        )
        stocks_sorted_list = sorted(list(universe.keys()))
    else:
        indexs = handler_data.index.levels[1].tolist()
        indexs = list(set(indexs))
        stocks_sorted_list = sorted(indexs)
    print("number of stocks: ", len(stocks_sorted_list))
    stocks_index_dict = {}
    for i, stock in enumerate(stocks_sorted_list):
        stocks_index_dict[stock] = i
    n = len(stocks_index_dict.keys())
    if relation_source == 'stock-stock':
        return stock_stock_data.get_all_matrix(
            market, stocks_index_dict,
            # data_path="D:/Code/myqlib/.qlib/qlib_data/graph_data/"
            data_path=os.path.join(args.data_root, 'graph_data')  # This is the data path on the server
        ), stocks_sorted_list
    elif relation_source == 'industry':
        industry_dict = stock_concept_data.read_graph_dict(
            market,
            relation_name="SW_belongs_to",
            # data_path="D:/Code/myqlib/.qlib/qlib_data/graph_data/",
            data_path=os.path.join(args.data_root, 'graph_data')
        )
        return stock_concept_data.get_full_connection_matrix(
            industry_dict, stocks_index_dict
        ), stocks_sorted_list
    elif relation_source == 'full':
        return np.ones(shape=(n, n)), stocks_sorted_list
    else:
        raise ValueError("unknown graph name `%s`" % relation_source)


def make_config(args):
    if args.market == "A_share":
        train_start_date = '2008-01-01'
        train_end_date = '2014-12-31'
        valid_start_date = '2015-01-01'
        valid_end_date = '2016-12-31'
        test_start_date = '2017-01-01'
        # explain_end_date = '2017-01-03'
        test_end_date = '2022-12-31'
        # test_end_date = explain_end_date
        explain_end_date = '2022-12-31'  # for each timestamp, attentionx requires about 3 sec to generate explanation.
    else:
        train_start_date = '2012-11-19'
        train_end_date = '2015-11-18'
        valid_start_date = '2015-11-19'
        valid_end_date = '2016-11-17'
        test_start_date = '2016-11-18'
        test_end_date = '2017-12-07'
        explain_end_date = '2016-12-18'  # for each timestamp, attentionx requires about 3 sec to generate explanation.

    market = args.market
    if market == "A_share":
        instrument = "csi300"
    elif market == "NYSE":
        instrument = "sp500"
    elif market == "NASDAQ":
        instrument = "nasdaq100"

    with open(
            '/data/zhaolifan/project/qlib/examples/benchmarks_dynamic/baseline/handler_mix_csi300_rankTrue_alpha360_horizon1.pkl',
            'rb') as f:
        handler = pickle.load(f)

    rel_encoding, stock_name_list = load_graph(
        market, instrument, train_start_date, test_end_date, args.relation_type, handler._data)

    config = {}
    config['model'] = {
        "class": "Graphs",
        "module_path": "qlib.contrib.model.pytorch_ts_dgl",
        "kwargs": {
            "graph_model": args.graph_model,  # 'GAT' or 'simpleHGN', 'RSR'
            "d_feat": 6,
            "hidden_size": 64,
            "num_layers": 1,
            "loss": "mse",
            "dropout": 0.7,
            "n_epochs": 100,
            "metric": "loss",
            "base_model": "LSTM",
            "use_residual": True,
            "GPU": args.gpu,
            "lr": 1e-4,
            "early_stop": 10,
            "rel_encoding": rel_encoding,
            "stock_name_list": stock_name_list,
            'num_graph_layer': 1
        },
    }
    config['log'] = {"class": "Graphs",
                     "module_path": "qlib.contrib.model.pytorch_ts_dgl",
                     "kwargs": {
                         "graph_model": args.graph_model,  # or 'simpleHGN'
                         "d_feat": 6,
                         "hidden_size": 64,
                         "num_layers": 1,
                         "loss": "mse",
                         "dropout": 0.7,
                         "n_epochs": 100,
                         "metric": "loss",
                         "base_model": "LSTM",
                         "use_residual": True,
                         "GPU": 0,
                         "lr": 1e-4,
                         "early_stop": 10,
                         "num_graph_layer": 1}}

    dh_config = {
        'start_time': train_start_date, 'end_time': test_end_date,
        'fit_start_time': train_start_date, 'fit_end_time': train_end_date,
        'infer_processors': [{'class': 'RobustZScoreNorm',
                              'kwargs': {'clip_outlier': True,
                                         'fields_group': 'feature'}},
                             {'class': 'Fillna',
                              'kwargs': {'fill_value': 0,
                                         'fields_group': 'feature'}}],
        'instruments': instrument,
        'label': ['Ref($close, -2) / Ref($close, -1) - 1'],
        'learn_processors': [{'class': 'DropnaLabel'},
                             {'class': 'CSRankNorm',
                              'kwargs': {'fields_group': 'label'}}],
    }

    # handler = {
    #     "class": "Alpha360",
    #     'module_path': 'qlib.contrib.data.handler',
    #     'kwargs': dh_config
    # }

    dataset_config = {
        'class': 'DatasetH',
        'module_path': 'qlib.data.dataset',
        'kwargs': {'handler': handler,
                   'segments': {
                       'train': (train_start_date, train_end_date),
                       'valid': (valid_start_date, valid_end_date),
                       'test': (test_start_date, test_end_date),
                       'explain': (test_start_date, explain_end_date)
                   },
                   }}
    # 'step_len':20}}
    config['dataset'] = dataset_config
    return config


def make_port_config(model, dataset, benchmark):
    return {
        "executor": {
            "class": "SimulatorExecutor",
            "module_path": "qlib.backtest.executor",
            "kwargs": {
                "time_per_step": "day",
                "generate_portfolio_metrics": True,
            },
        },
        "strategy": {
            "class": "TopkDropoutStrategy",
            "module_path": "qlib.contrib.strategy.signal_strategy",
            "kwargs": {
                "signal": (model, dataset),
                "topk": 50,
                "n_drop": 5,
            },
        },
        "backtest": {
            "start_time": "2017-01-01",
            "end_time": "2020-08-01",
            "account": 100000000,
            "benchmark": benchmark,
            "exchange_kwargs": {
                "freq": "day",
                "limit_threshold": 0.095,
                "deal_price": "close",
                "open_cost": 0.0005,
                "close_cost": 0.0015,
                "min_cost": 5,
            },
        },
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Explanation evalaation.")
    parser.add_argument("--data_root", type=str, default="/home/zhangzexi/.qlib/qlib_data/", help="data root path")
    parser.add_argument("--ckpt_root", type=str, default="/home/dengjiale/qlib_exp/tmp_ckpt/", help="ckpt root path")
    parser.add_argument("--result_root", type=str, default="/home/dengjiale/qlib_exp/results/",
                        help="explanation resluts root path")
    parser.add_argument("--market", type=str, default="A_share",
                        choices=["A_share", "NASDAQ", "NYSE"], help="market name")
    parser.add_argument("--relation_type", type=str, default="stock-stock",
                        choices=["stock-stock", "industry", "full"], help="relation type of graph")
    parser.add_argument("--graph_model", type=str, default="RSR",
                        choices=["RSR", "GAT", "simpleHGN"], help="graph moddel name")
    parser.add_argument("--graph_type", type=str, default="heterograph",
                        choices=["heterograph", "homograph"], help="graph type")
    parser.add_argument("--gpu", type=int, default=0, help="gpu number")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    device = torch.device("cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu")
    # use default data
    data_type = "cn_data" if args.market == "A_share" else "us_data"
    provider_uri = os.path.join(args.data_root, data_type)
    region = REG_CN if data_type == "cn_data" else REG_US

    GetData().qlib_data(target_dir=provider_uri, region=region, exists_skip=True)
    qlib.init(provider_uri=provider_uri, region=region)

    config = make_config(args)
    model = init_instance_by_config(config['model'])
    dataset = init_instance_by_config(config['dataset'])

    if args.market == "A_share":
        # instrument = "csi300"
        benchmark = "SH000300"
    elif args.market == "NASDAQ":
        benchmark = "^ndx"
    else:
        benchmark = "^gspc"
    port_analysis_config = make_port_config(model, dataset, benchmark)
    # NOTE: This line is optional
    # It demonstrates that the dataset can be used standalone.
    # example_df = dataset.prepare("train")
    # print(example_df.head())

    sparsity = {
        'RSR': {'effect': [3, 5, 9], 'xpath': [2, 3, 5], 'subgraphx': [1, 4, 8]},
        'simpleHGN': {'effect': [1, 3, 5], 'xpath': [2, 3, 5], 'subgraphx': [1, 4, 8]},
        'GAT': {'effect': [2, 3, 5], 'xpath': [2, 3, 5], 'subgraphx': [1, 3, 6]},
    }

    # start exp
    with R.start(experiment_name="graph_model_dgl"):
        R.log_params(**flatten_dict(config['log']))
        model_path = os.path.join(args.ckpt_root, f"{args.market}-{args.graph_model}-{args.graph_type}.pt")
        if os.path.exists(model_path):
            model.load_checkpoint(model_path)
            # R.save_objects(**{f"{args.market}-{args.graph_model}-{args.graph_type}.pkl": model})
        else:
            model.fit(dataset, save_path=model_path)

            # save_path = 'C:/Users/92553/tmp/tmpo820nos2'
            # model.load_checkpoint(save_path=save_path)

            # R.save_objects(**{f"{args.market}-{args.graph_model}-{args.graph_type}.pkl": model})

            # prediction
            recorder = R.get_recorder()
            sr = SignalRecord(model, dataset, recorder)
            sr.generate()

            # Signal Analysis
            sar = SigAnaRecord(recorder)
            sar.generate()

            # backtest. If users want to use backtest based on their own prediction,
            # please refer to https://qlib.readthedocs.io/en/latest/component/recorder.html#record-template.
            par = PortAnaRecord(recorder, port_analysis_config, "day")
            par.generate()


        def eval_explanation(explainer, explainer_name, step_size):
            print(f'=========={explainer_name}==========')
            if explainer_name == 'effect':
                ename = 'att'
            else:
                ename = explainer_name
            subgraphx_path = os.path.join(args.result_root,
                                          f"{args.market}-{args.graph_model}-{args.graph_type}-{ename}-explanation")
            if os.path.exists(subgraphx_path):
                with open(subgraphx_path, 'rb') as f:
                    attn_exp = pickle.load(f)
                for i, k in enumerate(sparsity[args.graph_model][explainer_name][1:]):
                    print(f'==================sparsity: {i + 3}====================')
                    explanation, scores = model.get_explanation(dataset, explainer,
                                                                cached_explanations=attn_exp['explanation'],
                                                                step_size=step_size,
                                                                top_k=k)
            else:
                print(subgraphx_path, 'does not exist!')
                explanation, scores = model.get_explanation(dataset, explainer)
                config['log']['explainer'] = explainer_name
                config['log']['explanation'] = explanation
                config['log']['scores'] = scores
                print('Saving explanations...')
                with open(subgraphx_path, 'wb') as f:
                    pickle.dump(config['log'], f)

        # get attention explanation
        graph_type = args.graph_type  # for GAT, use 'homograph'
        attn_explainer = AttentionX(graph_model=graph_type,
                                    num_layers=config['model']["kwargs"]['num_graph_layer'],
                                    device=device)
        # the num_layers of explainers decide the neighborhood in which to find the explanations,
        # usually its the same as
        xpath_explainer = xPath(graph_model=graph_type, num_layers=config['model']["kwargs"]['num_graph_layer'],
                                device=device)

        subagraphx_explainer = SubgraphXExplainer(graph_model=graph_type,
                                                  num_layers=config['model']["kwargs"]['num_graph_layer'],
                                                  device=device)

        eval_explanation(attn_explainer, 'effect', 100)
        eval_explanation(xpath_explainer, 'xpath', 100)
        eval_explanation(subagraphx_explainer, 'subgraphx', 200)

