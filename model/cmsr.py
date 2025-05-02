import torch
import torch.nn as nn
import torch.nn.init as init
from torch.nn.parameter import Parameter
from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.loss import BPRLoss
from recbole.model.init import xavier_normal_initialization
from recbole.data.interaction import Interaction

from model.encoder import Encoder
from model.layers import LayerNorm
from model.wce import calculate_item_popularity, calculate_weight, WCE
from model.wbpr import WBPRLoss


class CMSR(SequentialRecommender):

    def __init__(self, config, dataset, weight_dict=None):
        super(CMSR, self).__init__(config, dataset)


        # load dataset info
        self.item_num = dataset.item_num
        self.item_id = dataset.inter_feat.item_id

        # load parameters info
        self.num_layers = config['num_layers']
        self.num_heads = config['num_heads']
        self.embed_dim = config['embed_dim']
        self.ff_dim = config['ff_dim']
        self.hidden_dropout = config['hidden_dropout']
        self.attn_dropout = config['attn_dropout']
        self.layer_norm_eps = config['layer_norm_eps']
        self.activation = config['activation']
        self.initializer_range = config['initializer_range']
        # pretrain and finetune
        self.log_base = config['log_base']
        self.loss_type = config['loss_type']
        self.with_adapter = config['with_adapter']
        self.requires_grad = config['requires_grad']

        # load weights
        if weight_dict is not None:
            print("weight_dict is not None! Loading weights from the checkpoint...")
            self.w_q = self._load_weight('query', weight_dict, self.requires_grad)
            self.w_k = self._load_weight('key', weight_dict, self.requires_grad)
            self.w_v = self._load_weight('value', weight_dict, self.requires_grad)
        else:
            self.w_q = None
            self.w_k = None
            self.w_v = None

        # define layers
        # item_embedding -> [batch_size, seq_length, embedding_size]
        self.item_embedding = nn.Embedding.from_pretrained(dataset.item_feat.item_emb)
        # positional_embedding -> [batch_size, seq_length, embedding_size]
        self.embed_dim = dataset.item_feat.item_emb.shape[1]
        self.position_embedding = nn.Embedding(self.max_seq_length, self.embed_dim)
        self.cmsr_encoder = Encoder(
            num_layers=self.num_layers,
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            ff_dim=self.ff_dim,
            ffn_dict=None,
            w_q=self.w_q,
            w_k=self.w_k,
            w_v=self.w_v,
            with_mlp=self.with_adapter,
            hidden_dropout=self.hidden_dropout,
            attn_dropout=self.attn_dropout,
            layer_norm_eps=self.layer_norm_eps,
            activation=self.activation,
        )
        self.LayerNorm = LayerNorm(self.embed_dim, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout)


        # define loss
        if self.loss_type == 'BPR':
            self.loss = BPRLoss()
        elif self.loss_type == 'CE':
            self.loss = nn.CrossEntropyLoss()
        elif self.loss_type == 'WCE':  # weighted cross-entropy
            self.loss = WCE
            self.alpha = config['alpha']
            self.beta = config['beta']
            self.item_popularity_dict = calculate_item_popularity(self.item_num, self.item_id)
        elif self.loss_type == 'WBPR':  # weighted BPR
            self.loss = WBPRLoss()
            self.alpha = config['alpha']
            self.beta = config['beta']
            self.item_popularity_dict = calculate_item_popularity(self.item_num, self.item_id)
        else:
            raise NotImplementedError(
                "Make sure 'loss_type' in ['BPR', 'CE', 'WCE']!")

        # parameters initialization
        # self.apply(xavier_normal_initialization)
        # self.cmsr_encoder.apply(xavier_normal_initialization)
        self.apply(self._init_weights)
        self.cmsr_encoder.apply(self._init_weights)

        for name, param in self.cmsr_encoder.named_parameters():
            print(f"{name}: requires_grad={param.requires_grad}")
    
    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            if hasattr(module, 'weight') and module.weight.requires_grad:
                # 跳过已经加载的权重
                if getattr(self, "skip_pretrained_weights", False):
                    if (module.weight.data.shape == self.w_q.shape and torch.allclose(module.weight.data, self.w_q)) or \
                    (module.weight.data.shape == self.w_k.shape and torch.allclose(module.weight.data, self.w_k)) or \
                    (module.weight.data.shape == self.w_v.shape and torch.allclose(module.weight.data, self.w_v)):
                        return
                module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def _load_weight(self, weight_name, weight_dict, requires_grad):
            weight = weight_dict.get(weight_name)
            if weight is not None:
                param = Parameter(weight)
                if requires_grad:
                    param.requires_grad = False
                return param
            return None
    
    def _get_attention_mask(self, item_seq, bidirectional=False):
        """
        Generate left-to-right uni-directional or bidirectional attention mask for mha
        """
        attention_mask = item_seq != 0
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.bool
        if not bidirectional:
            extended_attention_mask = torch.tril(
                extended_attention_mask.expand((-1, -1, item_seq.size(-1), -1))
            )
        extended_attention_mask = torch.where(extended_attention_mask, 0.0, -10000.0)
        return extended_attention_mask
    
    def _gather_indexes(self, output, gather_index):
        """
        Gathers the vectors at the specific positions over a minibatch

        - gather_index: 
            Index of the last valid position of each sequence
        """
        gather_index = gather_index.view(-1, 1, 1).expand(-1, -1, output.shape[-1])
        output_tensor = output.gather(dim=1, index=gather_index)
        return output_tensor.squeeze(1)
    
    def forward(self, item_seq, item_seq_len):
        """
        Args:
            item_seq (torch.Tensor): item interaction sequence -> [batch_size, seq_len]
            item_seq_len (torch.Tensor): lengths of interaction sequences -> [batch_size]

        Returns:
            sequence representation -> [batch_size, hidden_size].
        """
        # item_seq: [batch_size, seq_len] elements: item ids(int)

        # position_ids: [seq_len], i.e. -> [0, 1, 2, ..., seq_len-1]
        # [seq_len] -> [batch_size, seq_len](copy for each batch)
        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        # [batch_size, seq_len] -> [batch_size, seq_len, embed_dim]
        position_embed = self.position_embedding(position_ids)
        
        # item_emb: [batch_size, seq_len, embed_dim] elements: item embeddings(float)
        item_emb = self.item_embedding(item_seq)

        # input_emb: item_emb + position_embed
        input_emb = self.LayerNorm(item_emb + position_embed)
        input_emb = self.dropout(input_emb)
       
        attn_mask = self._get_attention_mask(item_seq)
       
        
        cmsr_output = self.cmsr_encoder(
            input_emb, attn_mask
        )  # Shape: [batch_size, seq_len, hidden_size]
        # for name, param in self.cmsr_encoder.named_parameters():
        #     print(f"{name}: requires_grad={param.requires_grad}")

        output = self._gather_indexes(cmsr_output[-1], item_seq_len - 1)

        return output  # Shape: [batch_size, hidden_size]

    def calculate_loss(self, interaction):
        # print(interaction)

        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        pos_item = interaction[self.ITEM_ID]

        item_output = self.forward(item_seq, item_seq_len)  # [batch_size, hidden_size]
        # print("Item output:", item_output)

        if self.loss_type == 'BPR':
            neg_item = interaction[self.NEG_ITEM_ID]
            pos_item_embed = self.item_embedding(pos_item) # [batch_size, hidden_size]
            neg_item_embed = self.item_embedding(neg_item) # [batch_size, hidden_size]
            pos_item_score = torch.mul(item_output, pos_item_embed).sum(dim=1) # [batch_size]
            neg_item_score = torch.mul(item_output, neg_item_embed).sum(dim=1) # [batch_size]
            loss = self.loss(pos_item_score, neg_item_score)
        elif self.loss_type == 'CE':
            test_item_emb = self.item_embedding.weight  # [n_items, embed_dim]
            logits = torch.matmul(item_output, test_item_emb.transpose(0, 1))
            loss = self.loss(logits, pos_item)
        elif self.loss_type == 'WCE':
            test_item_emb = self.item_embedding.weight  # [n_items, embed_dim]
            logits = torch.matmul(item_output, test_item_emb.transpose(0, 1))
            # logits = torch.clamp(logits, min=-50, max=50)
            # print("Logits shape:", logits.shape)
            
            item_popularity = torch.tensor([self.item_popularity_dict[item.item()] for item in pos_item], 
                                           dtype=torch.float32, device=pos_item.device)
            # item_popularity = torch.clamp(item_popularity, min=0, max=500)
            weights = calculate_weight(item_popularity, self.alpha, self.beta)
            loss = self.loss(logits, pos_item, weights=weights)
        elif self.loss_type == 'WBPR':
            neg_item = interaction[self.NEG_ITEM_ID]
            pos_item_embed = self.item_embedding(pos_item)
            neg_item_embed = self.item_embedding(neg_item)
            pos_item_score = torch.mul(item_output, pos_item_embed).sum(dim=1)
            neg_item_score = torch.mul(item_output, neg_item_embed).sum(dim=1)
            pos_item_popularity = torch.tensor([self.item_popularity_dict[item.item()] for item in pos_item],
                                           dtype=torch.float32, device=pos_item.device)
            pos_weights = calculate_weight(pos_item_popularity, self.alpha, self.beta)
            neg_item_popularity = torch.tensor([self.item_popularity_dict[item.item()] for item in neg_item],
                                           dtype=torch.float32, device=neg_item.device)
            neg_weights = calculate_weight(neg_item_popularity, self.alpha, self.beta)
            loss = self.loss(pos_item_score, neg_item_score, pos_weights, neg_weights)

        if torch.isnan(loss).any():
            raise ValueError("Training loss is nan")
        
        # print("Loss:", loss.item())
        return loss

    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]
        test_item_embed = self.item_embedding(test_item)  # [batch_size, hidden_size]
        item_out = self.forward(item_seq, item_seq_len)  # [batch_size, hidden_size]

        scores = torch.mul(item_out, test_item_embed).sum(dim=1) # [batch_size]

        return scores

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        item_out = self.forward(item_seq, item_seq_len)
        all_item_embed = self.item_embedding.weight                   # [n_items, batch_size]

        scores = torch.matmul(item_out, all_item_embed.transpose(0, 1)) # [batch_size, n_items]

        return scores