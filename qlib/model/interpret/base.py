#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

"""
Interfaces to interpret models
"""

import pandas as pd
from abc import abstractmethod


class FeatureInt:
    """Feature (Int)erpreter"""

    @abstractmethod
    def get_feature_importance(self) -> pd.Series:
        """get feature importance

        Returns
        -------
            The index is the feature name.

            The greater the value, the higher importance.
        """


class LightGBMFInt(FeatureInt):
    """LightGBM (F)eature (Int)erpreter"""

    def __init__(self):
        self.model = None

    def get_feature_importance(self, *args, **kwargs) -> pd.Series:
        """get feature importance

        Notes
        -----
            parameters reference:
            https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.Booster.html?highlight=feature_importance#lightgbm.Booster.feature_importance
        """
        return pd.Series(
            self.model.feature_importance(*args, **kwargs), index=self.model.feature_name()
        ).sort_values(  # pylint: disable=E1101
            ascending=False
        )


class GraphExplainer:
    '''
    For 'rnn+gnn+predictor' model structure, this is an abstract interpreter to generate graph explanations.
    '''

    def __init__(self, graph_model, num_layers, device):
        self.graph_model = graph_model
        self.num_layers = num_layers
        self.device = device

    def explain(self, full_model, graph, stkid):
        return None

    def explaination_to_graph(self, explanation, subgraph, stkid): # form the explanations to a graph
        return subgraph, stkid