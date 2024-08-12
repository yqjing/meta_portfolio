import collections
import math

import torch

from torch import nn
from torch.nn import functional as F, init


def cosine(x1, x2, eps=1e-8):
    x1 = x1 / (torch.norm(x1, p=2, dim=-1, keepdim=True) + eps)
    x2 = x2 / (torch.norm(x2, p=2, dim=-1, keepdim=True) + eps)
    return x1 @ x2.transpose(0, 1)


# class LabelAdaptHead(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.weight = nn.Parameter(torch.empty(1))
#         self.bias = nn.Parameter(torch.ones(1) / 8)
#         init.uniform_(self.weight, 0.75, 1.25)
#
#     def forward(self, y, inverse=False):
#         if inverse:
#             return (y - self.bias) / (self.weight + 1e-9)
#         else:
#             return (self.weight + 1e-9) * y + self.bias

class LabelAdaptHeads(nn.Module):
    def __init__(self, num_head):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(1, num_head))
        self.bias = nn.Parameter(torch.ones(1, num_head) / 8)
        init.uniform_(self.weight, 0.75, 1.25)

    def forward(self, y, inverse=False):
        if inverse:
            return (y.view(-1, 1) - self.bias) / (self.weight + 1e-9)
        else:
            return (self.weight + 1e-9) * y.view(-1, 1) + self.bias

class LabelAdapter(nn.Module):
    def __init__(self, x_dim, num_head=4, temperature=4, hid_dim=32):
        super().__init__()
        self.num_head = num_head
        self.linear = nn.Linear(x_dim, hid_dim, bias=False)
        self.P = nn.Parameter(torch.empty(num_head, hid_dim))
        init.kaiming_uniform_(self.P, a=math.sqrt(5))
        # self.heads = nn.ModuleList([LabelAdaptHead() for _ in range(num_head)])
        self.heads = LabelAdaptHeads(num_head)
        self.temperature = temperature

    def forward(self, x, y, inverse=False):
        v = self.linear(x.reshape(len(x), -1))
        gate = cosine(v, self.P)
        gate = torch.softmax(gate / self.temperature, -1)
        # return sum([gate[:, i] * self.heads[i](y, inverse=inverse) for i in range(self.num_head)])
        return (gate * self.heads(y, inverse=inverse)).sum(-1)


class FiLM(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.scale = nn.Parameter(torch.empty(in_dim))
        nn.init.uniform_(self.scale, 0.75, 1.25)

    def forward(self, x):
        return x * self.scale


class FeatureAdapter(nn.Module):
    def __init__(self, in_dim, num_head=4, temperature=4):
        super().__init__()
        self.num_head = num_head
        self.P = nn.Parameter(torch.empty(num_head, in_dim))
        init.kaiming_uniform_(self.P, a=math.sqrt(5))
        self.heads = nn.ModuleList([nn.Linear(in_dim, in_dim, bias=True) for _ in range(num_head)])
        self.temperature = temperature

    def forward(self, x):
        s_hat = torch.cat(
            [torch.cosine_similarity(x, self.P[i], dim=-1).unsqueeze(-1) for i in range(self.num_head)], -1,
        )
        # s_hat = cosine(x, self.P)
        s = torch.softmax(s_hat / self.temperature, -1).unsqueeze(-1)
        return x + sum([s[..., i, :] * self.heads[i](x) for i in range(self.num_head)])


# asymmetric loss
class ASL(nn.Module):
    def __init__(self):
        super(ASL, self).__init__()

    def forward(self, y_hat, y_true):
        loss = torch.where(y_hat > y_true, (y_hat - y_true) ** 2, (y_hat - y_true) ** 2)
        return loss.mean()

class ForecastModel(nn.Module):
    def __init__(self, model: nn.Module, x_dim: int = None, lr: float = 0.001, weight_decay: float = 0,
                 need_permute: bool = False):
        """

        Args:
            model (nn.Module): the forecast model
            x_dim (int): the dimension of stock features (e.g., factor_num * time_series_length)
            lr (float): learning rate of forecast model
            weight_decay (float): L2 regularization of the (Adam) optimizer
            need_permute (bool): True when it requires time-series inputs to be shaped in [batch_size, factor_num * time_series_length] (e.g., in Qlib Alpha360)
        """
        super().__init__()
        self.lr = lr
        self.criterion = ASL()  # nn.MSELoss(), ASL()
        self.model = model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.need_permute = need_permute
        self.opt = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=weight_decay)
        if self.device is not None:
            self.to(self.device)

    def forward(self, X, model=None):
        """

        Args:
            X: [batch_size, x_dim]
            model: 

        Returns:
            predictions
        """
        if model is None:
            model = self.model
        if X.dim() == 3:
            X = X.permute(0, 2, 1).reshape(len(X), -1) if self.need_permute else X.reshape(len(X), -1)

        y_hat = model(X)
        y_hat = y_hat.view(-1)
        return y_hat


class DoubleAdapt(ForecastModel):
    def __init__(
        self, model, factor_num, x_dim=None, lr=0.001, weight_decay=0,
            need_permute=False, num_head=8, temperature=10,
    ):
        super().__init__(
            model, x_dim=x_dim, lr=lr, need_permute=need_permute, weight_decay=weight_decay,
        )
        self.teacher_x = FeatureAdapter(factor_num, num_head, temperature)
        self.teacher_y = LabelAdapter(factor_num if x_dim is None else x_dim, num_head, temperature)
        self.meta_params = list(self.teacher_x.parameters()) + list(self.teacher_y.parameters())
        if self.device is not None:
            self.to(self.device)

    def forward(self, X, model=None, transform=False):
        """

        Args:
            X: [batch_size, x_dim]
            model: a forecast model generated by MAML

        Returns:
            immediate predictions. If adapt_y is True, still need to transform y_hat in the outer space.
        """
        if transform:
            """ For a L-length time-series, X should be shaped in [batch_size, L, factor_num] """
            X = self.teacher_x(X)

        # # view the model
        # print("the model in forward is")
        # print(model)

        # # view the shape of X
        # print("the shape of X in the forward is", X.shape)
        
        return super().forward(X, model), X


class ALSTMModel(nn.Module):
    def __init__(self, d_feat=6, hidden_size=64, num_layers=2, dropout=0.0, rnn_type="GRU"):
        super().__init__()
        self.hid_size = hidden_size
        self.input_size = d_feat
        self.dropout = dropout
        self.rnn_type = rnn_type
        self.rnn_layer = num_layers
        self._build_model()

    def _build_model(self):
        try:
            klass = getattr(nn, self.rnn_type.upper())
        except Exception as e:
            raise ValueError("unknown rnn_type `%s`" % self.rnn_type) from e
        self.net = nn.Sequential()
        self.net.add_module("fc_in", nn.Linear(in_features=self.input_size, out_features=self.hid_size))
        self.net.add_module("act", nn.Tanh())
        self.rnn = klass(
            input_size=self.hid_size,
            hidden_size=self.hid_size,
            num_layers=self.rnn_layer,
            batch_first=True,
            dropout=self.dropout,
        )
        self.fc_out = nn.Linear(in_features=self.hid_size * 2, out_features=1)
        self.att_net = nn.Sequential()
        self.att_net.add_module(
            "att_fc_in",
            nn.Linear(in_features=self.hid_size, out_features=int(self.hid_size / 2)),
        )
        self.att_net.add_module("att_dropout", torch.nn.Dropout(self.dropout))
        self.att_net.add_module("att_act", nn.Tanh())
        self.att_net.add_module(
            "att_fc_out",
            nn.Linear(in_features=int(self.hid_size / 2), out_features=1, bias=False),
        )
        self.att_net.add_module("att_softmax", nn.Softmax(dim=1))

    def forward(self, inputs):
        # inputs: [batch_size, input_size*input_day]
        inputs = inputs.view(len(inputs), self.input_size, -1)
        inputs = inputs.permute(0, 2, 1)  # [batch, input_size, seq_len] -> [batch, seq_len, input_size]
        rnn_out, _ = self.rnn(self.net(inputs))  # [batch, seq_len, num_directions * hidden_size]
        attention_score = self.att_net(rnn_out)  # [batch, seq_len, 1]
        out_att = torch.mul(rnn_out, attention_score)
        out_att = torch.sum(out_att, dim=1)
        out = self.fc_out(
            torch.cat((rnn_out[:, -1, :], out_att), dim=1)
        )  # [batch, seq_len, num_directions * hidden_size] -> [batch, 1]
        # # =====apply tanh to normalize the output to [0, 1]======
        # out = torch.tanh(out)
        # out = (out + 1) / 2
        # # ====================================================
        return out[..., 0]


