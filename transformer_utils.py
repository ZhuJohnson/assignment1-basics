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
        #self.W = nn.Parameter(torch.empty(in_features, out_features, dtype=dtype, device=device))
        self.W = nn.Parameter(torch.empty(out_features, in_features, dtype=dtype, device=device))
        nn.init.trunc_normal_(self.W)

    def forward(self, x):
        
        return x @ (self.W.T)
    

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
        """Construct the RMSNorm module

        Args:
            d_model (int): Hidden dimension of the model
            eps (float, optional): Epsilon value for numerical stability. Defaults to 1e-5.
            device (None, optional): Device to store the parameters on. Defaults to None.
            dtype (None, optional): Data type of the parameters. Defaults to None.
        """
        super().__init__()
        self._d_model = d_model
        self._eps = eps
        self._device = dtype if dtype is not None else torch.device("cpu")
        self._dtype = dtype if dtype is not None else torch.float32
        self.W = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process an input tensor of shape (batch_size, sequence_length, d_model) and return a tensor of the same shape.

        Args:
            x (torch.Tensor): _description_

        Returns:
            torch.Tensor: _description_
        """
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms_val = 1 / torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True)+self._eps)
        res = x * rms_val * self.W
        return res.to(in_dtype)
    
def silu(x):
    return x * torch.sigmoid(x)

class SwiGLULayer(torch.nn.Module):
    def __init__(self, d_model, d_ff):
        """Construct the SwiGLU layer

        Args:
            d_model (int): Dimensionality of the feedforward input and output
            d_ff (int): Dimensionality of the up-project happening internally
        """
        super().__init__()
        self._d_model = d_model
        self._d_ff = d_ff
        self.w1 = nn.Parameter(torch.empty(d_ff, d_model)) #value weight
        self.w2 = nn.Parameter(torch.empty(d_model, d_ff)) #projection weight
        self.w3 = nn.Parameter(torch.empty(d_ff, d_model)) #gate weight
        nn.init.kaiming_normal_(self.w1)
        nn.init.xavier_normal_(self.w2)
        nn.init.kaiming_normal_(self.w3)
    
    def forward(self, x):
        return (silu(x @ self.w1.T) * (x @ self.w3.T)) @ self.w2.T


class RotaryPositionalEmbedding(torch.nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        """ Construct the RoPE module and create buffers if needed

        Args:
            theta (float): Θ value for the RoPE
            d_k (int): dimension of query and key vectors
            max_seq_len (int): Maximum sequence length that will be inputted
            device (None, optional): Device to store the buffer on. Defaults to None.
        """
        super().__init__()
        self._d_k = d_k
        self._theta = theta
        self._max_seq_len = max_seq_len
        self._device = torch.device("cpu") if device is None else device
        self.update_cache(max_seq_len)

    def update_cache(self, len: int):
        """when the sequence length is longer than max_seq_len which specified in this class,
        dynamically update the rotation matrix

        Args:
            len (int): the current sequence length
        """
        self._max_seq_len = len
        pos_multiplier = torch.arange(len, device=self._device).float()
        theta_list =  torch.pow(self._theta, -1 * torch.arange(0, self._d_k, 2)/self._d_k).float()
        #theta_{i,k} = i / (THETA ** (2k / d))
        #freq is theta_{i,k}, i between 0 and max_seq_len, k between 0, d_k/2
        freq = torch.einsum("i, j -> ij", pos_multiplier, theta_list)
        #shape of emb: [max_seq_len, d_k]
        #emb = torch.cat((freq,freq), dim=-1)
        emb = freq.repeat_interleave(2, dim=-1)
        self.register_buffer("sin", emb.sin(), persistent=False)
        self.register_buffer("cos", emb.cos(), persistent=False)

    def half_rotate(self, x):
        #(1,2,3,4) --> (-2,1,-4,3)
        x1 = x[..., 1::2] 
        x2 = x[..., ::2]   
        rotated = torch.stack((-x1, x2), dim=-1) 
        return rotated.flatten(start_dim=-2)      
    

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """Process an input tensor of shape (..., seq_len, d_k) and return a tensor of the same shape.

        Args:
            x (torch.Tensor): input tensor
            token_positions (torch.Tensor): position tensor

        Returns:
            torch.Tensor: tensor with rope 
        """
        max_len = torch.max(token_positions)
        if max_len >= self._max_seq_len:
            self.update_cache(max_len + 1)
            self._max_seq_len = max_len
        return x * self.cos[token_positions] + self.half_rotate(x) * self.sin[token_positions]
        
        