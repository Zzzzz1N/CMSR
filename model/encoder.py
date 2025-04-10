from torch import nn

from model.layers import EncoderLayer


class Encoder(nn.Module):
    """
    Encoder(Consist of multiple EncoderLayer)

    Args:
        - num_layers (int): number of encoder layers
        - embed_dim (int): dimension of model
        - num_heads (int): number of heads
        - ff_dim (int): dimension of feed-forward layer
        - ffn_dict (dict): dictionary containing weights and biases for the feed-forward layer
        - w_q/k/v (torch.Tensor): weight matrix for query/key/value
        - with_mlp (bool): whether to use MLP for query/key
        - hidden_dropout (float): dropout rate for hidden states
        - attn_dropout (float): dropout rate for attention scores
        - layer_norm_eps (float): epsilon for layer normalization
        - activation (str): activation function to use (default: 'gelu')
            Options: 'relu', 'gelu', 'swish', 'tanh', 'sigmoid'
    
    hidden_states -> [batch_size, seq_len, hidden_size]
    attn_mask -> [batch_size, 1, 1, seq_len] (optional)

    Returns:
        hidden_states (torch.Tensor): the output of the encoder layer

    """

    def __init__(
            self,
            num_layers = 2,
            embed_dim = 64,
            num_heads = 2,
            ff_dim = 256,
            ffn_dict = None,
            w_q = None,
            w_k = None,
            w_v = None,
            with_mlp = False,
            hidden_dropout = 0.5,
            attn_dropout = 0.5,
            layer_norm_eps = 1e-12,
            activation = "gelu",
        ):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList(
                        [EncoderLayer(embed_dim,
                                      num_heads,
                                      ff_dim,
                                      ffn_dict,
                                      w_q,
                                      w_k,
                                      w_v,
                                      with_mlp,
                                      hidden_dropout,
                                      attn_dropout,
                                      layer_norm_eps,
                                      activation)
                                      for _ in range(num_layers)])

    def forward(self, hidden_states, attn_mask, output_all_encoder_layer = True):
        """
        Returns:
            all_encoder_layers (list/torch.Tensor):
            output_all_encoder_layer -> True: return a list consists of all transformer layers' output
                                     -> False: return the output of last transformer layer
        """
        all_encoder_layers = []
        for layer in self.layers:
            hidden_states = layer(hidden_states, attn_mask)
            all_encoder_layers.append(hidden_states)

        return all_encoder_layers if output_all_encoder_layer else all_encoder_layers[-1]
