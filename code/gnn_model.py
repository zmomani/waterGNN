"""The reconstruction network: Chebyshev graph convolutions and a linear output.

The graph is fixed, so the Chebyshev terms are computed with one sparse matrix
that is shared by every sample in a batch. This is the same operation as
torch_geometric.nn.ChebConv (symmetric normalisation, largest eigenvalue taken
as 2) and runs several times faster on a processor.
"""
import torch


def scaled_laplacian(edge_index, n):
    """L~ = L_sym - I = -D^(-1/2) A D^(-1/2), as a sparse matrix."""
    row, col = edge_index
    deg = torch.bincount(row, minlength=n).float()
    w = -1.0 / torch.sqrt(deg[row] * deg[col])
    return torch.sparse_coo_tensor(edge_index, w, (n, n)).coalesce()


class ChebConv(torch.nn.Module):
    """Chebyshev graph convolution of order K (reaches K pipes away)."""
    def __init__(self, in_channels, out_channels, order):
        super().__init__()
        self.order = order
        self.lin = torch.nn.Linear(in_channels * (order + 1), out_channels)

    def forward(self, x, lap):                       # x: [batch, nodes, channels]
        b, n, c = x.shape
        t0 = x.permute(1, 0, 2).reshape(n, b * c)
        terms = [t0]
        if self.order > 0:
            terms.append(torch.sparse.mm(lap, t0))
        for _ in range(2, self.order + 1):
            terms.append(2 * torch.sparse.mm(lap, terms[-1]) - terms[-2])
        h = torch.stack(terms, dim=-1).reshape(n, b, c * (self.order + 1)).permute(1, 0, 2)
        return self.lin(h)


class Reconstructor(torch.nn.Module):
    def __init__(self, edge_index, n, in_channels=11, hidden=64, layers=4, order=10):
        super().__init__()
        self.register_buffer("lap", scaled_laplacian(edge_index, n).to_dense().to_sparse_csr(), persistent=False)
        dims = [in_channels] + [hidden] * layers
        self.convs = torch.nn.ModuleList(ChebConv(a, b, order) for a, b in zip(dims[:-1], dims[1:]))
        self.out = torch.nn.Linear(hidden, 1)

    def forward(self, x):                            # x: [batch, nodes, features]
        for conv in self.convs:
            x = torch.relu(conv(x, self.lap))
        return self.out(x).squeeze(-1)               # [batch, nodes]
