# Manual for Development & Experiments

## Development

Feel free to release your codes under `examples/`. Keep cautious about modifications on original codes of qlib. 

Upload new implementaions (e.g., some subclass that other developers can reuse) under `qlib/contrib`. 

Examples:

<table class="tg">
<thead>
  <tr>
    <th class="tg-9wq8" colspan="2">Level</th>
    <th class="tg-9wq8">Path of Base Class</th>
    <th class="tg-9wq8">Path of Subclass</th>
    <th class="tg-9wq8">Example</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-9wq8" rowspan="2">Data</td>
    <td class="tg-9wq8">Loading</td>
    <td class="tg-lboi">qlib/data/dataset.py</td>
    <td class="tg-lboi">qlib/contrib/data/dataset.py</td>
    <td class="tg-lboi">Loading knowledge graph</td>
  </tr>
  <tr>
    <td class="tg-9wq8">Processing</td>
    <td class="tg-lboi">qlib/data/processor.py</td>
    <td class="tg-lboi">qlib/contrib/data/processor.py</td>
    <td class="tg-lboi">Filling NaN of edge weights</td>
  </tr>
  <tr>
    <td class="tg-9wq8">Model</td>
    <td class="tg-9wq8">-</td>
    <td class="tg-lboi">qlib/model/base.py</td>
    <td class="tg-lboi">qlib/contrib/model/</td>
    <td class="tg-lboi">A proposed graph-based model</td>
  </tr>
  <tr>
    <td class="tg-9wq8" rowspan="3">Meta</td>
    <td class="tg-9wq8">Task</td>
    <td class="tg-lboi">qlib/model/meta/task.py</td>
    <td class="tg-lboi">qlib/contrib/meta/***/task.py</td>
    <td class="tg-lboi">A support set and a query set</td>
  </tr>
  <tr>
    <td class="tg-9wq8">Dataset</td>
    <td class="tg-lboi">qlib/model/meta/dataset.py</td>
    <td class="tg-lboi">qlib/contrib/meta/***/dataset.py</td>
    <td class="tg-lboi">A list of meta-train tasks</td>
  </tr>
  <tr>
    <td class="tg-9wq8">Meta-learner</td>
    <td class="tg-lboi">qlib/model/meta/model.py</td>
    <td class="tg-lboi">qlib/contrib/meta/***/model.py</td>
    <td class="tg-lboi">A meta-learner for the adjacency matrix</td>
  </tr>
  <tr>
    <td class="tg-9wq8">Interpretation</td>
    <td class="tg-9wq8">-</td>
    <td class="tg-lboi">qlib/model/interpret/base.py</td>
    <td class="tg-lboi">qlib/contrib/interpret/</td>
    <td class="tg-lboi">GNNExplainer</td>
  </tr>
  <tr>
    <td class="tg-9wq8">Ensembling</td>
    <td class="tg-9wq8">-</td>
    <td class="tg-lboi">qlib/model/ens/ensemble.py</td>
    <td class="tg-lboi">qlib/contrib/ens/ensemble.py</td>
    <td class="tg-lboi">Uniform weights</td>
  </tr>
</tbody>
</table>


## Metrics

**Regression metrics**

MSE, RMSE, MAE

> Different training algorithms lead to unobvious difference in errors (e.g., 0.0001).
>
> These metrics are more suitable for model selection.

**Ranking metrics**

IC, ICIR, Rank IC, Rank ICIR

> If we use CSRankNorm for label preprocessing, IC and ICIR are meaningless.

**Portfolio metrics**

Excess Annualized Return, Information Ratio, Max Drawdown

> All of them are based on `excess_return_with_cost` BUT NOT `without`.
>
> Max Drawdown cannot reflect the model performance and is not recommended.

**Other metrics**

Precision@K

> Not recommended.

## Data

### Data preparation

```bash
# Yahoo Finance data of A-share market
python scripts/get_data.py qlib_data --target_dir ~/.qlib/qlib_data/cn_data --region cn

# crowd-source version data of A-share market
wget https://github.com/chenditc/investment_data/releases/download/20220720/qlib_bin.tar.gz
tar -zxvf qlib_bin.tar.gz -C ~/.qlib/qlib_data/crowd_data --strip-components=2

# US-Stock data
python scripts/get_data.py qlib_data --target_dir ~/.qlib/qlib_data/us_data --region us

```

The crowd-source data is unnecessary if we do not compare with DDG-DA.

Alternative: [converting csv format into qlib format](https://qlib.readthedocs.io/en/latest/component/data.html#converting-csv-format-into-qlib-format).

### Stock market

Qlib's hyperparameters are tuned on CSI300 while the performance is poor on US-Stock data.

| Stock Set | Benchmark |
| --------- | --------- |
| csi300    | SH000300  |
| csi500    | SH000905  |
| sp500     | ^gspc     |
| nasdaq100 | ^ndx      |

```python
if data_dir == 'us_data':
    if market == 'sp500':
        benchmark = "^gspc"
    elif market == 'nasdaq100':
        benchmark = "^ndx"
elif market == 'csi500':
    benchmark = "SH000905"
elif market == 'csi300':
    benchmark = "SH000300"
```

Also revise the instrument and the label definition after loading yaml files.

```python
conf['task']['dataset']['kwargs']['handler']['kwargs']['instruments'] = market
if data_dir == 'us_data':
    conf["task"]["dataset"]["kwargs"]["handler"]["kwargs"]["label"] = [
        "Ref($close, -{}) / $close - 1".format(horizon)
    ]
```

### Data preprocessing

#### Alpha

Choose feature format according to your model type, *e.g.*, Alpha360 for time-series models and Alpha158 for GBDT and MLP.

#### Label normalization

Most existing studies normalize the labels grouped by date. 

REST and HIST use `qlib.data.dataset.processor.CSZscoreNorm`, while DDG-DA uses `qlib.data.dataset.processor.CSRankNorm`.

## Codes

#### Qlib initialization

```python
if data_dir == 'cn_data':
    auto_init()
else:
    qlib.init(provider_uri='~/.qlib/qlib_data/' + data_dir, region='us' if self.data_dir == 'us_data' else 'cn')
```

Field `region` is IMPORTANT for backtesting!

#### Backtesting

```python
def backtest(self, exp_name, pred_y_all):
    backtest_config = {
        "strategy": {
            "class": "TopkDropoutStrategy",
            "module_path": "qlib.contrib.strategy",
            "kwargs": {"signal": "<PRED>", "topk": 50, "n_drop": 5},
        },
        "backtest": {
            "start_time": None,
            "end_time": None,
            "account": 100000000,
            "benchmark": benchmark,
            "exchange_kwargs": {
                "limit_threshold": None if data_dir == 'us_data' else 0.095,
                "deal_price": "close",
                "open_cost": 0.0005,
                "close_cost": 0.0015,
                "min_cost": 5,
            },
        },
    }
    rec = R.get_exp(experiment_name=exp_name).list_recorders(rtype=Experiment.RT_L)[0]
    rmse = np.sqrt(((pred_y_all['pred'].to_numpy() - pred_y_all['label'].to_numpy()) ** 2).mean())
    mae = np.abs(pred_y_all['pred'].to_numpy() - pred_y_all['label'].to_numpy()).mean()
    print('rmse:', rmse, 'mae', mae)
    rec.log_metrics(rmse=rmse, mae=mae)
    SigAnaRecord(recorder=rec, skip_existing=True).generate()
    PortAnaRecord(recorder=rec, config=backtest_config, skip_existing=True).generate()
    return rec
```

