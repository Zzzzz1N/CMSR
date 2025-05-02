import torch
import torch.nn as nn


class MLPAdapter(nn.Module):
    """
    MLP Adapter for q, k matrices in Transformer
    
    Args:
        - d_model (int): Dimension of the model (default: 768)
        - adapter_dim (int): Dimension of the adapter (default: 256)
        - dropout (float): Dropout rate (default: 0.1)
    """
    # def __init__(self, d_model=768, adapter_dim=256, dropout=0.1):
    #     super().__init__()
    #     self.d_model = d_model
    #     self.adapter_dim = adapter_dim
        
    #     # MLP structure: Linear -> GELU -> Dropout -> Linear -> Dropout
    #     self.adapter = nn.Sequential(
    #         nn.Linear(d_model, adapter_dim),
    #         nn.GELU(),
    #         nn.Dropout(dropout),
    #         nn.Linear(adapter_dim, d_model),
    #         nn.Dropout(dropout)
    #     )
        
    #     self.scale = nn.Parameter(torch.ones(1))
    def __init__(self, d_model=768, adapter_dim=256, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.adapter_dim = adapter_dim
        
        # MLP structure: Linear -> GELU -> Dropout -> Linear -> Dropout
        self.adapter = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.ReLU()
        )
        
        self.scale = nn.Parameter(torch.ones(1))
        
    def forward(self, x):
        x = x + self.scale * self.adapter(x)
        return x


# class MLP(nn.Module):
#     """
#     MLP Layers

#     Args:
#         - input_dim (int): Input dimension
#         - hidden_dim (int): Hidden dimension
#         - output_dim (int): Output dimension
#         - dropout (float): Dropout rate (default: 0)
#         - activation (str): Activation function to use (default: 'relu')
#             Options: 'sigmoid', 'tanh', 'relu', 'leakyrelu', 'gelu'
    
#     Example:
#         >>> model = MLP(input_dim=10, hidden_dim=20, output_dim=5, dropout=0.5, activation='relu')
#         >>> input = torch.randn(32, 10)  # Batch of 32 samples with 10 features each
#         >>> output = model(input)  # torch.Size([32, 5])
#     """
#     def __init__(self, input_dim, hidden_dim, output_dim, dropout=0, activation='relu'):
#         super(MLP, self).__init__()
#         self.input_dim = input_dim
#         self.hidden_dim = hidden_dim
#         self.output_dim = output_dim
#         self.dropout = dropout
#         self.activation = activation

#         self.dropout_layer = nn.Dropout(dropout) if dropout > 0 else None
#         self.fc1 = nn.Linear(input_dim, hidden_dim)
#         self.activation_layer = self._get_activation_func(self.activation)
#         self.fc2 = nn.Linear(hidden_dim, output_dim)
#         # self.fc2 = nn.Linear(hidden_dim, hidden_dim)
#         # self.fc3 = nn.Linear(hidden_dim, output_dim)

#     def forward(self, x):
#         if self.dropout_layer:
#             x = self.dropout_layer(x)
#         x = self.fc1(x)
#         if self.activation_layer:
#             x = self.activation_layer(x)
#         x = self.fc2(x)
#         if self.activation_layer:
#             x = self.activation_layer(x)
#         # x = self.fc3(x)
#         return x
    
#     def _get_activation_func(self, act_name):
#         if act_name is None:
#             return None
#         ACT2FN = {
#             "sigmoid": nn.Sigmoid(),
#             "tanh": nn.Tanh(),
#             "relu": nn.ReLU(),
#             "leakyrelu": nn.LeakyReLU(negative_slope=0.01),
#             "gelu": nn.GELU(),
#         }
#         return ACT2FN[act_name]

