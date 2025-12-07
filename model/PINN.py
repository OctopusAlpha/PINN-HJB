import torch.nn as nn


class PINN(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super().__init__()
        layers = []
        dims = [input_dim] + hidden_dims + [output_dim]
        for i in range(len(dims)-2):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.nn = nn.Sequential(*layers)

    def forward(self, x):
        # output = self.nn(x)
        # V = torch.nn.functional.relu(output[:, 0:1])  # 或用 torch.relu
        # raw_pi = output[:, 1:]
        # return torch.cat([V, raw_pi], dim=1)
        return self.nn(x)  # 输出：V + π

