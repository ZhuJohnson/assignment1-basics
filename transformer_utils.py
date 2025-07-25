import torch
from torch import nn
import math
import einops

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
        if token_positions is None:
            token_positions = torch.tensor(range(x.shape[-2]))
        return x * self.cos[token_positions] + self.half_rotate(x) * self.sin[token_positions]
        
def safe_softmax(x, softmax_dim):
    """apply softmax to the i-th dimension of the input
    tensor x. The output tensor should have the same shape as the input tensor, but its i-th dimension will
    now have a normalized probability distribution.

    Args:
        x (torch.Tensor): input tensor
        softmax_dim (int): the dimension which the softmax will apply on
    """
    max_val = torch.max(x, dim=softmax_dim, keepdim=True).values
    logits = torch.exp(x - max_val)
    logits_sum = logits.sum(dim=softmax_dim, keepdim=True)
    return logits / logits_sum

def scaled_dot_product_attention(q, k, v, mask=None):
    """calculate the scaled dot product consuming q/k/v/mask

    Args:
        q (torch.tensor): (batch_size, ..., seq_len_q, d_k)
        k (torch.Tensor): (batch_size, ..., seq_len_k, d_k)
        v (torch.Tensor): (batch_size, ..., seq_len_k, d_v)
        mask (torch.Tensor, optional): the boolean torch tensor, value=True means corresponding query attend to keys

    Returns:
        _type_: _description_
    """
    d_k = q.shape[-1]
    x = torch.einsum("... a d, ... b d -> ... a b", q, k) / d_k ** 0.5
    if mask is not None:
        x = x.masked_fill(~mask, float('-inf'))
    y = safe_softmax(x, -1)
    output = torch.einsum("... a b, ... b c -> ... a c", y, v)
    return output

class MultiHead_self_Attention(torch.nn.Module):
    def __init__(self, d_model, num_heads, rope: nn.Module = None, device = None, dtype = None):
        """ causal multi-head self-attention implementation

        Args:
            d_model (int): Dimensionality of the Transformer block inputs.
            num_heads (int): Number of heads to use in multi-head self-attention.
            rope (nn.Module): the optional rope transformation
            device
            dtype
        """
        super().__init__()
        self._d_model = d_model
        self._num_heads = num_heads
        self._rope = rope
        self._d_k, self._d_v = d_model // num_heads, d_model // num_heads
        self._qkv_layer = LinearLayer(d_model, 3*d_model, device=device, dtype=dtype)
        #self._q_layer = LinearLayer(d_model, d_model, device=device, dtype=dtype)
        #self._k_layer = LinearLayer(d_model, d_model, device=device, dtype=dtype)
        #self._v_layer = LinearLayer(d_model, d_model, device=device, dtype=dtype)
        self._wo_layer = LinearLayer(d_model, d_model, device=device, dtype=dtype)
    
    def forward(self, x, token_positions=None):
        seq_len = x.shape[-2]
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device = x.device, dtype = torch.bool))
        QKV = self._qkv_layer(x)
        Q, K, V = QKV.chunk(3, dim=-1)
        #Q = self._q_layer(x)
        #K = self._k_layer(x)
        #V = self._v_layer(x)
        Q = einops.rearrange(Q, "... seq_len (num_heads d_head) -> ... num_heads seq_len d_head", num_heads = self._num_heads, d_head = self._d_k)
        K = einops.rearrange(K, "... seq_len (num_heads d_head) -> ... num_heads seq_len d_head", num_heads = self._num_heads, d_head = self._d_k)
        V = einops.rearrange(V, "... seq_len (num_heads d_head) -> ... num_heads seq_len d_head", num_heads = self._num_heads, d_head = self._d_v)
        if self._rope:
            Q = self._rope(Q, token_positions)
            K = self._rope(K, token_positions)

        attn = scaled_dot_product_attention(Q, K, V, mask = causal_mask)
        attn = einops.rearrange(attn, "... num_heads seq_len dv -> ... seq_len (num_heads dv)", num_heads = self._num_heads, dv=self._d_v)
        return self._wo_layer(attn)