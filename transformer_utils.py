import torch
from torch import nn

class LinearLayer(torch.nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        """Construct a linear transformation module

        Args:
            in_features (int): final dimension of the input
            out_features (int): final dimension of the output
            device (None, optional): Device to store the parameters on. Defaults to None.
            dtype (None, optional): Data type of the parameters. Defaults to None.
        """
        super().__init__()

        self._in_features = in_features
        self._out_features = out_features
        self._device = device
        self._dtype = dtype if dtype is not None else torch.float32
        #self.W = torch.empty(in_features, out_features, dtype=dtype, device=device)
        self.W = nn.Parameter(torch.empty(in_features, out_features, dtype=dtype, device=device))
        nn.init.trunc_normal_(self.W)

    def forward(self, x):
        
        return x @ self.W
    

class EmbeddingLayer(torch.nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        """_summary_

        Args:
            num_embeddings (int): Size of the vocabulary
            embedding_dim (int): Dimension of the embedding vectors
            device (None, optional): Device to store the parameters on. Defaults to None.
            dtype (None, optional): Data type of the parameters. Defaults to None.
        """
        super().__init__()
        self._num_embeddings = num_embeddings
        self._embedding_dim = embedding_dim
        self._device = dtype if dtype is not None else torch.device("cpu")
        self._dtype = dtype if dtype is not None else torch.float32
        self.W = nn.Parameter(torch.empty(num_embeddings, embedding_dim, device = self._device, dtype = self._dtype))
        nn.init.trunc_normal_(self.W)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.W[token_ids]
    

class RMSNormLayer(torch.nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        """_summary_

        Args:
            d_model (int): Hidden dimension of the model
            eps (float, optional): Epsilon value for numerical stability. Defaults to 1e-5.
            device (None, optional): Device to store the parameters on. Defaults to None.
            dtype (None, optional): Data type of the parameters. Defaults to None.
        """