from __future__ import absolute_import, division

import math
import torch
import torch.nn as nn


class GSNetGraphConv(nn.Module):
    def __init__(self, in_features, out_features, adj, beta=0.2, bias=True):
        super(GSNetGraphConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.beta = beta

        self.W = nn.Parameter(
            torch.zeros(size=(2, in_features, out_features), dtype=torch.float)
        )
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        self.M = nn.Parameter(
            torch.zeros(size=(adj.size(1), out_features), dtype=torch.float)
        )
        nn.init.xavier_uniform_(self.M.data, gain=1.414)

        self.A = adj

        self.Q = nn.Parameter(torch.ones_like(adj))
        nn.init.constant_(self.Q, 1e-6)

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.float))
            stdv = 1.0 / math.sqrt(self.W.size(2))
            self.bias.data.uniform_(-stdv, stdv)
        else:
            self.register_parameter("bias", None)

    def forward(self, input, X):
        # Feature Transformation
        HW1 = torch.matmul(input, self.W[0])
        HW2 = torch.matmul(input, self.W[1])

        # Adjacency Modulation
        A = self.beta * self.A.to(input.device) + self.beta * self.Q.to(input.device)

        A_upper = torch.triu(A, diagonal=1).to(input.device)
        A_lower = A_upper.T

        # Symmetry Regularization
        A_upper = (A_upper + A_upper.T) / 2
        A_lower = (A_lower + A_lower.T) / 2

        I = torch.eye(A.size(0), dtype=torch.float).to(input.device)

        first_term_temp = torch.matmul(((1 - self.beta) * I + A_lower), A_upper)

        # Weght Modulation
        first_term = torch.matmul(first_term_temp * I, self.M * HW1) + torch.matmul(
            first_term_temp * (1 - I), self.M * HW2
        )

        XW = torch.matmul(X, self.W[0])

        # Weght Modulation
        second_term = torch.matmul(
            (1 - self.beta) * I + A_lower, self.M * XW
        )

        output = first_term + second_term

        if self.bias is not None:
            return output + self.bias.view(1, 1, -1)
        else:
            return output


    def __repr__(self):
        return (
            self.__class__.__name__
            + " ("
            + str(self.in_features)
            + " -> "
            + str(self.out_features)
            + ")"
        )
