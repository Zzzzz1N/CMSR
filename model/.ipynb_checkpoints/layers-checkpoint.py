import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math

from model.adapter import MLPAdapter


class LayerNorm(nn.Module):
    def __init__(self, embed_dim, eps=1e-12):
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(embed_dim))
        self.bias = nn.Parameter(torch.zeros(embed_dim))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, unbiased=False, keepdim=True)
        normalized = (x - mean) / torch.sqrt(var + self.eps)
        normalized = self.weight * normalized + self.bias
        return normalized
    
class MultiHeadAttention(nn.Module):
    """
    Multi-head self-attention layer

    Args:
        - embed_dim (int): dimension of model
        - n_heads (int): number of heads
        - w_q/k/v (torch.Tensor): weight matrix for query/key/value
        - with_mlp (bool): whether to use MLP for query/key
        - attn_dropout (float): dropout rate for attention scores
    
    input_tensor -> [batch_size, seq_len, hidden_size]

    Returns:
        hidden_states (torch.Tensor): the output of the MHA layer

    """

    def __init__(self, embed_dim, num_heads, w_q=None, w_k=None, w_v=None, 
                 with_mlp=False, attn_dropout=0):

        super(MultiHeadAttention, self).__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.w_q = nn.Linear(embed_dim, embed_dim)
        self.w_k = nn.Linear(embed_dim, embed_dim) 
        self.w_v = nn.Linear(embed_dim, embed_dim)

        self.with_mlp = with_mlp
        if with_mlp:
            # self.mlp_q = MLP(embed_dim, (embed_dim//3)*4, embed_dim, dropout=0.1, activation='leakyrelu')
            # self.mlp_k = MLP(embed_dim, (embed_dim//3)*4, embed_dim, dropout=0.1, activation='leakyrelu')
            self.mlp_q = MLPAdapter(d_model=embed_dim, dropout=0.1)
            self.mlp_k = MLPAdapter(d_model=embed_dim, dropout=0.1)
        
        if w_q is not None:
            self.w_q.weight = w_q
        if w_k is not None:
            self.w_k.weight = w_k
        if w_v is not None:
            self.w_v.weight = w_v
        
        self.softmax = nn.Softmax(dim=-1)
        self.attn_dropout = nn.Dropout(attn_dropout)
        
        self.w_out = nn.Linear(embed_dim, embed_dim)

    def forward(self, input_tensor, attn_mask=None):
        batch_size, seq_len, _ = input_tensor.size()

        q, k, v = self.w_q(input_tensor), self.w_k(input_tensor), self.w_v(input_tensor)

        if self.with_mlp:
            q = self.mlp_q(q)
            k = self.mlp_k(k)

        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        # [batch_size, seq_len, hidden_size] -> [batch_size, n_head, seq_len, attn_head_size]

        attn_score = torch.matmul(q, k.transpose(-1, -2)) / (self.head_dim ** 0.5)

        if attn_mask is not None:
            # attn_score = attn_score.masked_fill(attn_mask == 0, float("-inf"))
            attn_score = attn_score + attn_mask
        
        attn_score = self.softmax(attn_score)
        attn_score = self.attn_dropout(attn_score)

        attn_output = torch.matmul(attn_score, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        # [batch_size, n_head, seq_len, attn_head_size] -> [batch_size, seq_len, hidden_size]

        output = self.w_out(attn_output)
        
        return output

class FeedForward(nn.Module):
    """
    Feed-forward layer 

    Args:
        - embed_dim (int): dimension of model
        - ff_dim (int): dimension of feed-forward layer
        - ffn_dict (dict): dictionary containing weights and biases for the feed-forward layer
        - activation (str): activation function to use
            Options: 'relu', 'gelu', 'swish', 'tanh', 'sigmoid'
    """
    def __init__(self, embed_dim, ff_dim, ffn_dict, activation='relu'):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(embed_dim, ff_dim)
        self.activation_layer = self.get_activation_func(activation)
        self.linear2 = nn.Linear(ff_dim, embed_dim)

        # Load stored weights if provided
        if ffn_dict is not None:
            self.linear1.weight = Parameter(ffn_dict.get('layer1_weight'))
            self.linear1.bias = Parameter(ffn_dict.get('layer1_bias'))
            # self.linear1.weight.requires_grad = False
            # self.linear1.bias.requires_grad = False

            self.linear2.weight = Parameter(ffn_dict.get('layer2_weight'))
            self.linear2.bias = Parameter(ffn_dict.get('layer2_bias'))
            # self.linear2.weight.requires_grad = False
            # self.linear2.bias.requires_grad = False

    def forward(self, input_tensor):
        input_tensor = self.linear1(input_tensor)
        input_tensor = self.activation_layer(input_tensor)

        input_tensor = self.linear2(input_tensor)

        return input_tensor
    
    def get_activation_func(self, act):
        ACT2FN = {
            "gelu": self.gelu,
            "relu": F.relu,
            "swish": self.swish,
            "tanh": F.tanh,
            "sigmoid": F.sigmoid,
        }
        return ACT2FN[act]
    
    def gelu(self, x):
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

    def swish(self, x):
        return x * torch.sigmoid(x)

class EncoderLayer(nn.Module):

    """
    Encoder Layer

    Returns:
        hidden_states (torch.Tensor): the output of the encoder layer

    """

    def __init__(self, embed_dim, num_heads, ff_dim, ffn_dict, w_q, w_k, w_v, with_mlp, 
                 hidden_dropout, attn_dropout, layer_norm_eps, activation):
        super(EncoderLayer, self).__init__()

        self.attention = MultiHeadAttention(embed_dim, num_heads, w_q=w_q, w_k=w_k, w_v=w_v, 
                                            with_mlp=with_mlp, attn_dropout=attn_dropout)
        self.dropout1 = nn.Dropout(hidden_dropout)
        self.norm1 = LayerNorm(embed_dim, eps=layer_norm_eps)
        
        self.ffn = FeedForward(embed_dim, ff_dim, ffn_dict, activation=activation)
        self.dropout2 = nn.Dropout(hidden_dropout)
        self.norm2 = LayerNorm(embed_dim, eps=layer_norm_eps)

    def forward(self, x, attn_mask):
        _x = x
        x = self.attention(x, attn_mask)
        
        x = self.dropout1(x)
        x = self.norm1(x + _x)
        
        _x = x
        x = self.ffn(x)
      
        x = self.dropout2(x)
        x = self.norm2(x + _x)
        return x







# if __name__ == "__main__":
#     # Example usage
#     embed_dim = 64
#     num_heads = 8
#     batch_size = 32
#     seq_len = 10

#     mha = MultiHeadAttention(embed_dim, num_heads)
#     input_tensor = torch.randn(batch_size, seq_len, embed_dim)
#     output_tensor = mha(input_tensor)

#     print("Output shape:", output_tensor.shape)  # Should be [batch_size, seq_len, embed_dim]