import random
import numpy as np
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GINConv, global_mean_pool

from .modules import CompGCNConv, PrototypeLearning, PrototypePrediction,\
    BiAttentionGNNConv, \
    TimeGapEmbedding, \
    Aggregator

from pyhealth.models import RNNLayer, TransformerLayer, GRASPLayer, StageNetLayer, AdaCareLayer, DeeprLayer


class Deepr(nn.Module):
    
    def __init__(self, args):
        super().__init__()

        device = args['device']
        self.num_nodes = args['num_nodes']
        self.visit_thresh = args['visit_thresh']
        self.max_weeks_between = args['max_weeks_between']
        self.visit_code_num = args['visit_code_num']
        self.start_embed_dim = args['start_embed_dim']
        self.hidden_dim = args['hidden_dim']
        self.window_size = args['window_size']
        self.out_dim = args['out_dim']
        node_embed = args['global_node_attr']

        self.full_node_ids = torch.arange(self.num_nodes).to(device)

        if node_embed is None:
            self.node_embed = nn.Embedding(self.num_nodes, self.start_embed_dim, padding_idx=0)
        else:
            self.node_embed = nn.Embedding.from_pretrained(node_embed, freeze=False, padding_idx=0)
            self.node_projection = nn.Linear(node_embed.size(-1), self.start_embed_dim)

        self.time_gap_embed = TimeGapEmbedding(self.start_embed_dim, device)

        self.layer = DeeprLayer(feature_size=self.start_embed_dim, window=self.window_size, hidden_size=self.hidden_dim)

        self.head = nn.Linear(self.hidden_dim, self.out_dim)

    def forward(self, node_ids, visit_times, attn_mask, **kwargs):

        B, V, N = node_ids.shape
        x = self.node_embed(self.full_node_ids)
        x = self.node_projection(x)

        x = x.unsqueeze(0).unsqueeze(0).expand(B, V, -1, -1)  # (B,V,node_num,D)
        x = torch.gather(x, 2, node_ids.unsqueeze(-1).expand(-1, -1, -1, self.start_embed_dim))  # (B,V,N,D)

        inv_attn_mask = ~attn_mask
        mean_mask = inv_attn_mask.unsqueeze(-1).expand(-1, -1, -1, x.size(-1)).to(int)
        x = x * mean_mask
        x = torch.sum(x, dim=2)
        mean_count = inv_attn_mask.to(int).sum(dim=-1, keepdim=True)
        mean_count[mean_count == 0] = 1
        x = x / mean_count  # (B,V,D)

        visit_mask = attn_mask.all(dim=-1)
        visit_mask = ~visit_mask
        visit_mask = visit_mask.to(int)

        x = self.layer(x, visit_mask)  # (B,D)

        output = self.head(x)

        return {'logits': [output], 'prototypes': None, 'embeddings': None}


class AdaCare(nn.Module):
    
    def __init__(self, args):
        super().__init__()

        self.device = args['device']
        self.num_nodes = args['num_nodes']
        self.visit_thresh = args['visit_thresh']
        self.visit_code_num = args['visit_code_num']
        self.input_dim = args['input_dim']
        self.hidden_dim = args['hidden_dim']
        self.kernel_size = args['kernel_size']
        self.kernel_num = args['kernel_num']
        self.out_dim = args['out_dim']
        self.r_v = args['r_v']
        self.r_c = args['r_c']
        self.activation = args['activation']
        node_embed = args['global_node_attr']
        self.dropout = 0.

        self.full_node_ids = torch.arange(self.num_nodes).to(self.device)

        if node_embed is None:
            self.node_embed = nn.Embedding(self.num_nodes, self.input_dim, padding_idx=0)
        else:
            self.node_embed = nn.Embedding.from_pretrained(node_embed, freeze=False, padding_idx=0)
            self.node_projection = nn.Linear(node_embed.size(-1), self.input_dim)

        self.adacare = AdaCareLayer(input_dim=self.input_dim,
                                    hidden_dim=self.hidden_dim,
                                    kernel_size=self.kernel_size,
                                    kernel_num=self.kernel_num,
                                    r_v=self.r_v,
                                    r_c=self.r_c,
                                    dropout=0.1)

        self.head = nn.Linear(self.hidden_dim, self.out_dim)

    def forward(self, node_ids, visit_times, attn_mask, **kwargs):

        B, V, N = node_ids.shape
        x = self.node_embed(self.full_node_ids)
        x = self.node_projection(x)

        x = x.unsqueeze(0).unsqueeze(0).expand(B, V, -1, -1)  # (B,V,node_num,D)
        x = torch.gather(x, 2, node_ids.unsqueeze(-1).expand(-1, -1, -1, self.input_dim))

        inv_attn_mask = ~attn_mask
        mean_mask = inv_attn_mask.unsqueeze(-1).expand(-1, -1, -1, x.size(-1)).to(int)
        x = x * mean_mask
        x = torch.sum(x, dim=2)
        mean_count = inv_attn_mask.to(int).sum(dim=-1, keepdim=True)
        mean_count[mean_count == 0] = 1
        x = x / mean_count  # (B,V,D)

        visit_mask = attn_mask.all(dim=-1)
        inv_visit_mask = ~visit_mask
        mask = inv_visit_mask.to(int)

        x, _, _, _ = self.adacare(x, mask)

        output = self.head(x)

        return {'logits': [output], 'prototypes': None, 'embeddings': None}


class StageNet(nn.Module):
    
    def __init__(self, args):
        super().__init__()

        self.device = args['device']
        self.num_nodes = args['num_nodes']
        self.visit_thresh = args['visit_thresh']
        self.visit_code_num = args['visit_code_num']
        self.input_dim = args['input_dim']
        self.hidden_dim = args['hidden_dim']
        self.conv_dim = args['hidden_dim']
        self.conv_size = args['conv_size']
        self.output_dim = args['out_dim']
        self.levels = args['levels']
        node_embed = args['global_node_attr']
        self.chunk_size = self.hidden_dim // self.levels
        assert self.hidden_dim % self.levels == 0

        self.full_node_ids = torch.arange(self.num_nodes).to(self.device)

        if node_embed is None:
            self.node_embed = nn.Embedding(self.num_nodes, self.input_dim, padding_idx=0)
        else:
            self.node_embed = nn.Embedding.from_pretrained(node_embed, freeze=False, padding_idx=0)
            self.node_projection = nn.Linear(node_embed.size(-1), self.input_dim)

        self.stagenet = StageNetLayer(input_dim=self.input_dim,
                                      chunk_size=self.hidden_dim,
                                      conv_size=self.conv_size,
                                      levels=self.levels,
                                      dropconnect=0.,
                                      dropout=0.1,
                                      dropres=0.)

        self.head = nn.Linear(self.hidden_dim * self.levels, self.output_dim)

    def forward(self, node_ids, visit_times, attn_mask, **kwargs):

        B, V, N = node_ids.shape
        x = self.node_embed(self.full_node_ids)
        x = self.node_projection(x)

        x = x.unsqueeze(0).unsqueeze(0).expand(B, V, -1, -1)  # (B,V,node_num,D)
        x = torch.gather(x, 2, node_ids.unsqueeze(-1).expand(-1, -1, -1, self.input_dim))

        inv_attn_mask = ~attn_mask
        mean_mask = inv_attn_mask.unsqueeze(-1).expand(-1, -1, -1, x.size(-1)).to(int)
        x = x * mean_mask
        x = torch.sum(x, dim=2)
        mean_count = inv_attn_mask.to(int).sum(dim=-1, keepdim=True)
        mean_count[mean_count == 0] = 1
        x = x / mean_count  # (B,V,D)

        visit_mask = attn_mask.all(dim=-1)
        inv_visit_mask = ~visit_mask
        mask = inv_visit_mask.to(int)

        x, _, _ = self.stagenet(x, time=visit_times, mask=mask)

        output = self.head(x)

        return {'logits': [output], 'prototypes': None, 'embeddings': None}


class GRASP(nn.Module):
    
    def __init__(self, args):
        super().__init__()

        self.device = args['device']
        self.num_nodes = args['num_nodes']
        self.num_edges = args['num_edges']
        self.num_visits = args['visit_thresh']
        self.num_codes = args['visit_code_num']
        self.out_dim = args['out_dim']
        self.start_embed_dim = args['start_embed_dim']
        self.hidden_dim = args['hidden_dim']
        self.cluster_num = args['cluster_num']
        self.block = args['block']
        node_embed = args['global_node_attr']

        self.full_node_ids = torch.arange(self.num_nodes).to(self.device)

        if node_embed is None:
            self.node_embed = nn.Embedding(self.num_nodes, self.start_embed_dim, padding_idx=0)
        else:
            self.node_embed = nn.Embedding.from_pretrained(node_embed, freeze=False, padding_idx=0)
            self.node_projection = nn.Linear(node_embed.size(-1), self.start_embed_dim)

        self.grasp = GRASPLayer(input_dim=self.hidden_dim,
                                static_dim=0,
                                hidden_dim=self.hidden_dim,
                                cluster_num=self.cluster_num,
                                block=self.block,
                                dropout=0.1)

        self.head = nn.Linear(self.hidden_dim, self.out_dim)

    def forward(self, node_ids, visit_times, attn_mask, **kwargs):

        x = self.node_embed(self.full_node_ids)  # (node_num,D)
        x = self.node_projection(x)

        B, V, N = node_ids.shape
        x = x.unsqueeze(0).unsqueeze(0).expand(B, V, -1, -1)  # (B,V,node_num,D)
        x = torch.gather(x, 2, node_ids.unsqueeze(-1).expand(-1, -1, -1, self.hidden_dim))

        inv_attn_mask = ~attn_mask
        mean_mask = inv_attn_mask.unsqueeze(-1).expand(-1, -1, -1, x.size(-1)).to(int)
        x = x * mean_mask
        x = torch.sum(x, dim=2)
        mean_count = inv_attn_mask.to(int).sum(dim=-1, keepdim=True)
        mean_count[mean_count == 0] = 1
        x = x / mean_count  # (B,V,D)

        visit_mask = attn_mask.all(dim=-1)
        inv_visit_mask = ~visit_mask
        inv_visit_mask = inv_visit_mask.to(int)

        x = self.grasp(x, mask=inv_visit_mask)  # (B,D)

        out = self.head(x)

        return {'logits': [out], 'prototypes': None, 'embeddings': None}


class KerPrint(nn.Module):

    def __init__(self, args):
        super().__init__()

        self.device = args['device']
        self.num_nodes = args['num_nodes']
        self.num_edges = args['num_edges']
        self.num_visits = args['visit_thresh']
        self.num_codes = args['visit_code_num']
        self.out_dim = args['out_dim']
        self.start_embed_dim = args['start_embed_dim']
        self.gnn_layer = args['gnn_layer']
        self.hidden_dim = args['hidden_dim']
        self.head_num = args['head_num']
        self.encoder_depth = args['encoder_depth']
        self.dropout = args['dropout']
        node_embed = args['global_node_attr']
        edge_embed = None
        edge_index = args['global_edge_index']
        edge_ids = args['global_edge_ids']
        edge_type = args['global_edge_type']

        unique_edge_num = len(torch.unique(edge_type))
        self.edge_type = edge_type.to(self.device)
        self.full_node_ids = torch.arange(self.num_nodes).to(self.device)
        self.full_edge_ids = torch.arange(unique_edge_num).to(self.device)

        if node_embed is None:
            self.node_embed = nn.Embedding(self.num_nodes, self.start_embed_dim, padding_idx=0)
            self.node_projection = None
        else:
            self.node_embed = nn.Embedding.from_pretrained(node_embed, freeze=False, padding_idx=0)
            self.node_projection = nn.Linear(node_embed.size(-1), self.start_embed_dim)

        if edge_embed is None:
            self.edge_embed = nn.Embedding(unique_edge_num, self.start_embed_dim, padding_idx=0)
            self.edge_projection = None
        else:
            edge_embed = edge_embed.to(torch.float32)
            self.edge_embed = nn.Embedding.from_pretrained(edge_embed, freeze=False, padding_idx=0)
            self.edge_projection = nn.Linear(edge_embed.size(-1), self.start_embed_dim)

        self.edge_index = edge_index.to(self.device)
        self.edge_ids = edge_ids


        self.kgat_dim_list = [self.start_embed_dim] + self.gnn_layer * [self.hidden_dim]
        self.n_layers_KGAT = len(self.kgat_dim_list)

        self.takgat_dim_list = [self.start_embed_dim] + self.gnn_layer * [self.hidden_dim]
        self.n_layers_TAKGAT = len(self.takgat_dim_list)

        self.kgat_hidden_dim = np.sum(np.array(self.kgat_dim_list))
        self.Transformer_Encoder = TransformerLayer(feature_size=self.kgat_hidden_dim,
                                            heads=self.head_num,
                                            dropout=self.dropout,
                                            num_layers=self.encoder_depth)

        self.element_weight_matrix = nn.Parameter(torch.Tensor(self.num_nodes, self.hidden_dim, self.hidden_dim))
        self.patient_transform_fc = nn.Linear(self.kgat_hidden_dim, self.hidden_dim)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.dropout_layer = nn.Dropout(self.dropout)
        self._init_weight()

        self.aggregator_layers_TAKGAT = nn.ModuleList()
        for k in range(self.n_layers_TAKGAT - 1):
            self.aggregator_layers_TAKGAT.append(Aggregator(self.takgat_dim_list[k], self.takgat_dim_list[k + 1], self.dropout))

        self.aggregator_layers_KGAT = nn.ModuleList()
        for k in range(self.n_layers_KGAT - 1):
            self.aggregator_layers_KGAT.append(Aggregator(self.kgat_dim_list[k], self.kgat_dim_list[k + 1], self.dropout))

        self.classfier_fc = nn.Linear(self.hidden_dim + self.kgat_hidden_dim, self.out_dim)

    def _init_weight(self):
        nn.init.xavier_uniform_(self.edge_embed.weight)

    def calc_knowledge_embed(self, sequence_embedding_final, canknowledge_embed):
        
        candidate_knowledge_num = canknowledge_embed.size(1)
        patient_element_matrix = sequence_embedding_final.unsqueeze(2).expand(-1, -1, sequence_embedding_final.size(1))
        patient_element_matrix_expand = patient_element_matrix.unsqueeze(1).expand(-1, candidate_knowledge_num, -1, -1)
        knowledge_element_matrix = canknowledge_embed.unsqueeze(2).expand(-1, -1, sequence_embedding_final.size(1), -1)
        interaction_matrix = patient_element_matrix_expand * knowledge_element_matrix + patient_element_matrix_expand - knowledge_element_matrix
        element_attention_matrix = torch.softmax(torch.sum(interaction_matrix * self.element_weight_matrix, dim=2), dim=1)
        result_embedding = torch.sum(canknowledge_embed * element_attention_matrix, dim=1)
        
        return result_embedding, element_attention_matrix


    def calc_sequence_embedding(self, g_global, g_personal, node_attr, node_ids, attn_mask):

        g_global = g_global.local_var()
        g_personal = g_personal.local_var()

        ego_embed = node_attr
        all_embed = [ego_embed]

        for i,layer in enumerate(self.aggregator_layers_TAKGAT):
            ego_embed = layer(g_personal, ego_embed)
            norm_embed = F.normalize(ego_embed, p=2, dim=1)
            all_embed.append(norm_embed)
        all_embed = torch.cat(all_embed, dim=1)

        B, V, N = node_ids.shape
        x = all_embed.unsqueeze(0).unsqueeze(0).expand(B, V, -1, -1)  # (B,V,node_num,D)
        x = torch.gather(x, 2, node_ids.unsqueeze(-1).expand(-1, -1, -1, self.kgat_hidden_dim))  # (B,V,N,D)

        inv_attn_mask = ~attn_mask
        mean_mask = inv_attn_mask.unsqueeze(-1).expand(-1, -1, -1, x.size(-1)).to(int)
        x = x * mean_mask
        x = torch.sum(x, dim=2)
        mean_count = inv_attn_mask.to(int).sum(dim=-1, keepdim=True)
        mean_count[mean_count == 0] = 1
        x = x / mean_count  

        visit_mask = attn_mask.all(dim=-1)
        visit_mask = ~visit_mask
        visit_mask = visit_mask.to(int)    

        _, sequence_embedding_final = self.Transformer_Encoder(x, visit_mask)   # (B,D)
          
        global_node_embed = node_attr
        kgat_embed = []

        for i, layer in enumerate(self.aggregator_layers_KGAT):
            global_node_embed = layer(g_global, global_node_embed)
            norm_embed = F.normalize(global_node_embed, p=2, dim=1)
            kgat_embed = [norm_embed]
        kgat_embed = torch.cat(kgat_embed, dim=0)

        canknowledge_embed = kgat_embed.expand(B, kgat_embed.size(0), kgat_embed.size(1))
        sequence_embedding_transform = self.relu(self.patient_transform_fc(sequence_embedding_final))
        knowledge_embed, _ = self.calc_knowledge_embed(sequence_embedding_transform,canknowledge_embed)

        sequence_embedding_all = torch.cat((sequence_embedding_final, knowledge_embed),dim=1)
        sequence_embedding_all = self.dropout_layer(sequence_embedding_all)

        return sequence_embedding_all


    def forward(self, node_ids, visit_times, attn_mask, **kwargs):

        node_attr = self.node_embed(self.full_node_ids)  # (node_num,D)
        edge_attr = self.edge_embed(self.full_edge_ids)

        if self.node_projection is not None:
            node_attr = self.node_projection(node_attr)
        if self.edge_projection is not None:
            edge_attr = self.edge_projection(edge_attr)

        edge_attr = edge_attr[self.edge_type]

        g_global = dgl.graph((self.edge_index[0], self.edge_index[1]))
        g_global.ndata['node'] = node_attr
        g_global.edata['att'] = edge_attr
        g_global = g_global.to(self.device)

        batch_node_ids = torch.unique(node_ids.flatten())
        g_personal = dgl.node_subgraph(g_global, batch_node_ids, relabel_nodes=False)

        all_sequence_embed = self.calc_sequence_embedding(g_global, g_personal, node_attr, node_ids, attn_mask)
          
        out = self.classfier_fc(all_sequence_embed)

        return {'logits': [out], 'prototypes': None, 'embeddings': None}


class GraphCare(nn.Module):
    
    def __init__(self, args):
        super().__init__()

        self.gnn = args['gnn']
        self.layer = args['layer']
        self.embedding_dim = args['start_embed_dim']
        self.decay_rate = args['decay_rate']
        self.patient_mode = args['patient_mode']
        self.use_alpha = args['use_alpha']
        self.use_beta = args['use_beta']
        self.edge_attn = args['edge_attn']
        self.num_nodes = args['num_nodes']
        self.num_edges = args['num_edges']
        self.max_visit = args['visit_thresh']
        self.hidden_dim = args['hidden_dim']
        self.out_channels = args['out_dim']
        node_embed = args['global_node_attr']
        edge_embed = None
        self.drop_rate = 0.
        self.dropout = args['dropout']
        self.attn_init = args['attn_init']
        self.self_attn = 0.

        j = torch.arange(self.max_visit).float()
        self.lambda_j = torch.exp(self.decay_rate * (self.max_visit - j)).unsqueeze(0).reshape(1, self.max_visit,
                                                                                               1).float()

        if node_embed is None:
            self.node_emb = nn.Embedding(self.num_nodes, self.embedding_dim, padding_idx=0)
        else:
            self.node_emb = nn.Embedding.from_pretrained(node_embed, freeze=False, padding_idx=0)

        if edge_embed is None:
            self.edge_emb = nn.Embedding(self.num_edges, self.embedding_dim, padding_idx=0)
        else:
            edge_emb = edge_embed.to(torch.float32)
            self.edge_emb = nn.Embedding.from_pretrained(edge_emb, freeze=False, padding_idx=0)

        self.lin_node = nn.Linear(768, self.hidden_dim)
        self.lin_edge = nn.Linear(self.embedding_dim, self.hidden_dim)
        self.bn1 = nn.BatchNorm1d(self.hidden_dim)

        self.alpha_attn = nn.ModuleDict()
        self.beta_attn = nn.ModuleDict()
        self.conv = nn.ModuleDict()
        self.bn_gnn = nn.ModuleDict()

        self.leakyrelu = nn.LeakyReLU(0.1)
        self.relu = nn.ReLU()
        self.tahh = nn.Tanh()

        for layer in range(1, self.layer + 1):
            if self.use_alpha:
                self.alpha_attn[str(layer)] = nn.Linear(self.num_nodes, self.num_nodes)

                if self.attn_init is not None:
                    attn_init_matrix = torch.eye(
                        self.num_nodes).float() * self.attn_init  # Multiply the identity matrix by attn_init
                    self.alpha_attn[str(layer)].weight.data.copy_(
                        attn_init_matrix)  # Copy the modified attn_init_matrix to the weights

                else:
                    nn.init.xavier_normal_(self.alpha_attn[str(layer)].weight)
            if self.use_beta:
                self.beta_attn[str(layer)] = nn.Linear(self.num_nodes, 1)
                nn.init.xavier_normal_(self.beta_attn[str(layer)].weight)
            if self.gnn == "BAT":
                self.conv[str(layer)] = BiAttentionGNNConv(nn.Linear(self.hidden_dim, self.hidden_dim),
                                                           edge_dim=self.hidden_dim,
                                                           edge_attn=self.edge_attn,
                                                           eps=self.self_attn)
            elif self.gnn == "GAT":
                self.conv[str(layer)] = GATConv(self.hidden_dim, self.hidden_dim)
            elif self.gnn == "GIN":
                self.conv[str(layer)] = GINConv(nn.Linear(self.hidden_dim, self.hidden_dim))

        if self.patient_mode == "joint":
            self.MLP = nn.Linear(self.hidden_dim * 2, self.out_channels)
        else:
            self.MLP = nn.Linear(self.hidden_dim, self.out_channels)

    def to(self, device):
        
        super().to(device)
        self.lambda_j = self.lambda_j.float().to(device)
        return self

    def forward(self, node_ids, visit_times, attn_mask, in_drop=False, store_attn=False, **kwargs):

        node_ids = kwargs['cat_node_ids']
        edge_ids = kwargs['cat_edge_ids']
        edge_index = kwargs['cat_edge_index']
        visit_node = kwargs['visit_nodes']  # (B,V,node_num)
        ehr_nodes = kwargs['ehr_nodes']  # (B,node_num)
        batch = kwargs['batch']

        if in_drop and self.drop_rate > 0:
            edge_count = edge_index.size(1)
            edges_to_remove = int(edge_count * self.drop_rate)
            indices_to_remove = set(random.sample(range(edge_count), edges_to_remove))
            edge_index = edge_index[:,
                                    [i for i in range(edge_count) if i not in indices_to_remove]].to(edge_index.device)
            edge_ids = torch.tensor([rel_id for i, rel_id in enumerate(edge_ids) if i not in indices_to_remove],
                                    device=edge_ids.device)

        x = self.node_emb(node_ids).float()
        edge_attr = self.edge_emb(edge_ids).float()

        x = self.lin_node(x)
        edge_attr = self.lin_edge(edge_attr)

        if store_attn:
            self.alpha_weights = []
            self.beta_weights = []
            self.attention_weights = []
            self.edge_weights = []

        for layer in range(1, self.layer + 1):
            if self.use_alpha:
                alpha = torch.softmax((self.alpha_attn[str(layer)](visit_node.float())),
                                      dim=1)  # (batch, max_visit, num_nodes)

            if self.use_beta:
                beta = torch.tanh(
                    (self.beta_attn[str(layer)](visit_node.float()))) * self.lambda_j  # (batch, max_visit, 1)

            if self.use_alpha and self.use_beta:
                attn = alpha * beta  # (batch, max_visit, num_nodes)
            elif self.use_alpha:
                attn = alpha * torch.ones((batch.max().item() + 1, self.max_visit, 1)).to(edge_index.device)
            elif self.use_beta:
                attn = beta * torch.ones((batch.max().item() + 1, self.max_visit, self.num_nodes)).to(edge_index.device)
            else:
                attn = torch.ones((batch.max().item() + 1, self.max_visit, self.num_nodes)).to(edge_index.device)

            attn = torch.sum(attn, dim=1)
            xj_node_ids = node_ids[edge_index[0]]
            xj_batch = batch[edge_index[0]]
            attn = attn[xj_batch, xj_node_ids].reshape(-1, 1)

            if self.gnn == "BAT":
                x, w_rel = self.conv[str(layer)](x, edge_index, edge_attr, attn=attn)
            else:
                x = self.conv[str(layer)](x, edge_index)

            x = F.relu(x)
            x = F.dropout(x, p=0.5, training=self.training)

            if store_attn:
                self.alpha_weights.append(alpha)
                self.beta_weights.append(beta)
                self.attention_weights.append(attn)
                self.edge_weights.append(w_rel)

        if self.patient_mode == "joint" or self.patient_mode == "graph":
            x_graph = global_mean_pool(x, batch)  # (B,D)
            x_graph = F.dropout(x_graph, p=self.dropout, training=self.training)

        if self.patient_mode == "joint" or self.patient_mode == "node":
            # patient node embedding through local (direct EHR) mean pooling
            x_node = torch.stack([
                ehr_nodes[i].view(1, -1) @ self.node_emb.weight / torch.sum(ehr_nodes[i])
                for i in range(batch.max().item() + 1)
            ])

            x_node = self.lin_node(x_node).squeeze(1)
            x_node = F.dropout(x_node, p=self.dropout, training=self.training)  # (B,D)

        if self.patient_mode == "joint":
            # concatenate patient graph embedding and patient node embedding
            x_concat = torch.cat((x_graph, x_node), dim=1)
            x_concat = F.dropout(x_concat, p=self.dropout, training=self.training)

            # MLP for prediction
            logits = self.MLP(x_concat)

        elif self.patient_mode == "graph":
            # MLP for prediction
            logits = self.MLP(x_graph)

        elif self.patient_mode == "node":
            # MLP for prediction
            logits = self.MLP(x_node)

        return {'logits': [logits], 'prototypes': None, 'embeddings': None}


class ProtoEHR(nn.Module):
    
    def __init__(self, args):
        super().__init__()

        self.device = args['device']
        self.num_nodes = args['num_nodes']
        self.num_edges = args['num_edges']
        self.num_visits = args['visit_thresh']
        self.num_codes = args['visit_code_num']
        self.out_dim = args['out_dim']
        self.start_embed_dim = args['start_embed_dim']
        self.gnn_type = args['gnn_type']
        self.gnn_layer = args['gnn_layer']
        self.hidden_dim = args['hidden_dim']
        self.ff_dim = args['ff_dim']
        self.head_num = args['head_num']
        self.depth = args['depth']
        self.dropout = args['dropout']
        self.max_weeks_between = args['max_weeks_between']
        node_embed = args['global_node_attr']
        edge_embed = None
        edge_index = args['global_edge_index']
        edge_ids = args['global_edge_ids']
        edge_type = args['global_edge_type']
        edge_norm = args['global_edge_norm']
        graph = args['graph']

        unique_edge_num = len(torch.unique(edge_type))
        self.full_node_ids = torch.arange(self.num_nodes).to(self.device)
        self.full_edge_ids = torch.arange(unique_edge_num).to(self.device)

        self.edge_type = edge_type.to(self.device)
        self.edge_norm = edge_norm.to(self.device)
        self.graph = graph.to(self.device)
        self.node_projection = None
        self.edge_projection = None

        self.patient_proto_num = args['patient_proto_num']
        self.visit_proto_num = args['visit_proto_num']
        self.code_proto_num = args['code_proto_num']
        self.softmax_temp = args['softmax_temp']
        self.attn_iter = args['attn_iter']

        if node_embed is None:
            self.node_embed = nn.Embedding(self.num_nodes, self.start_embed_dim, padding_idx=0)
        else:
            self.node_embed = nn.Embedding.from_pretrained(node_embed, freeze=False, padding_idx=0)
            self.node_projection = nn.Linear(node_embed.size(-1), self.start_embed_dim)

        if edge_embed is None:
            self.edge_embed = nn.Embedding(unique_edge_num, self.start_embed_dim, padding_idx=0)
        else:
            edge_embed = edge_embed.to(torch.float32)
            self.edge_embed = nn.Embedding.from_pretrained(edge_embed, freeze=False, padding_idx=0)
            self.edge_projection = nn.Linear(edge_embed.size(-1), self.start_embed_dim)

        self.edge_index = edge_index.to(self.device)
        self.edge_ids = edge_ids

        if self.gnn_type == 'GAT':
            self.gnn = nn.ModuleList([GATConv(self.hidden_dim, self.hidden_dim) for _ in range(self.gnn_layer)])
        elif self.gnn_type == 'CompGCN':
            self.gnn = nn.ModuleList([
                CompGCNConv(self.hidden_dim, self.hidden_dim, torch.tanh, drop_rate=self.dropout, opn='rotate')
                for _ in range(self.gnn_layer)
            ])

        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=self.hidden_dim,
                                       nhead=self.head_num,
                                       dim_feedforward=self.ff_dim,
                                       dropout=self.dropout,
                                       batch_first=True), self.depth)

        self.patient_PtL = PrototypeLearning(prototype_num=self.patient_proto_num,
                                          hidden_dim=self.hidden_dim,
                                          attn_iter=self.attn_iter)
        
        self.visit_PtL = PrototypeLearning(prototype_num=self.visit_proto_num,
                                          hidden_dim=self.hidden_dim,
                                          attn_iter=self.attn_iter)
        
        self.code_PtL = PrototypeLearning(prototype_num=self.code_proto_num,
                                          hidden_dim=self.hidden_dim,
                                          attn_iter=self.attn_iter)

        self.patient_PtP = PrototypePrediction(prototype_num=self.patient_proto_num,
                                               hidden_dim=self.hidden_dim,
                                               out_dim=self.out_dim)
        
        self.visit_PtP = PrototypePrediction(prototype_num=self.visit_proto_num,
                                               hidden_dim=self.hidden_dim,
                                               out_dim=self.out_dim)
        
        self.code_PtP = PrototypePrediction(prototype_num=self.code_proto_num,
                                               hidden_dim=self.hidden_dim,
                                               out_dim=self.out_dim)
        
        self.code_PtF = nn.MultiheadAttention(embed_dim=self.hidden_dim,
                                              num_heads=1,
                                              dropout=self.dropout,
                                              batch_first=True)
        
        self.visit_PtF = nn.MultiheadAttention(embed_dim=self.hidden_dim,
                                              num_heads=1,
                                              dropout=self.dropout,
                                              batch_first=True)
        
        self.patient_PtF = nn.MultiheadAttention(embed_dim=self.hidden_dim,
                                              num_heads=1,
                                              dropout=self.dropout,
                                              batch_first=True)
        
        self.PtF_projection = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.PtF_weights = nn.Parameter(torch.Tensor(self.hidden_dim, 1))
        nn.init.xavier_normal_(self.PtF_weights)

        self.head = nn.Linear(self.hidden_dim, self.out_dim)
        
    def prototype_fusion(self, embedding, code_prototypes, visit_prototypes, patient_prototypes):
        
        x_code, code_attn_weights = self.code_PtF(embedding, code_prototypes, code_prototypes)
        x_visit, visit_attn_weights = self.visit_PtF(embedding, visit_prototypes, visit_prototypes)
        x_patient, patient_attn_weights = self.patient_PtF(embedding, patient_prototypes, patient_prototypes)

        code_weight = torch.matmul(x_code, self.PtF_weights)
        visit_weight = torch.matmul(x_visit, self.PtF_weights)
        patient_weight = torch.matmul(x_patient, self.PtF_weights)
        weights = torch.concat((code_weight, visit_weight, patient_weight), dim=-1)
        weights = torch.softmax(weights/self.softmax_temp, dim=-1).unsqueeze(dim=1) # (B,3,1)

        x_code = x_code.unsqueeze(dim=1)  # (B,1,D)
        x_visit = x_visit.unsqueeze(dim=1)  # (B,1,D)
        x_patient = x_patient.unsqueeze(dim=1)  # (B,1,D)
        x = torch.concat((x_code, x_visit, x_patient), dim=1)   # (B,3,D)
        x = torch.matmul(weights, x).squeeze(dim=1)

        return x, weights.squeeze(dim=1), (code_attn_weights, visit_attn_weights, patient_attn_weights)

    def forward(self, node_ids, visit_times, attn_mask, **kwargs):
        
        x = self.node_embed(self.full_node_ids)  # (node_num,D)
        edge_attr = self.edge_embed(self.full_edge_ids)  # (edge_num,D)

        if self.node_projection is not None:
            x = self.node_projection(x)
        if self.edge_projection is not None:
            edge_attr = self.edge_projection(edge_attr)

        original_code_embeddings = x.clone()

        # Global GNN
        if self.gnn_type == 'GAT':
            for layer in self.gnn:
                x = layer(x, self.edge_index, edge_attr=edge_attr)
        elif self.gnn_type == 'CompGCN':
            for layer in self.gnn:
                x, edge_attr = layer(self.graph, x, edge_attr, self.edge_type, self.edge_norm)

        _, _, code_prototypes, code_PtL_weights = self.code_PtL(x)
        code_PtL_weights = code_PtL_weights.T

        # Select
        B, V, N = node_ids.shape
        x = x.unsqueeze(0).unsqueeze(0).expand(B, V, -1, -1)  # (B,V,node_num,D)
        x = torch.gather(x, 2, node_ids.unsqueeze(-1).expand(-1, -1, -1, self.hidden_dim))  # (B,V,N,D)
        
        x = x.reshape(-1, x.size(-1))
        x_code = self.code_PtP(x, code_prototypes)
        x_code = x_code.reshape(B, V, N, x_code.size(-1))
        x = x.reshape(B, V, N, x.size(-1))
        x = x + x_code

        # AvgPool
        inv_attn_mask = ~attn_mask
        mean_mask = inv_attn_mask.unsqueeze(-1).expand(-1, -1, -1, x.size(-1)).to(int)
        x = x * mean_mask
        x = torch.sum(x, dim=2)
        mean_count = inv_attn_mask.to(int).sum(dim=-1, keepdim=True)
        mean_count[mean_count == 0] = 1
        x = x / mean_count  # (B,V,D)

        visit_mask = attn_mask.all(dim=-1)
        visit_proto_mask = visit_mask.flatten()
        visit_proto_x = x.reshape(-1, x.size(-1))
        visit_proto_x = visit_proto_x[~visit_proto_mask]

        _, _, visit_prototypes, visit_PtL_weights = self.visit_PtL(visit_proto_x)
        x = x.reshape(-1, x.size(-1))
        x_visit = self.visit_PtP(x, visit_prototypes)
        x_visit = x_visit.reshape(B, V, x_visit.size(-1))
        x = x.reshape(B, V, x.size(-1))
        x = x + x_visit

        count_mask = ~visit_mask
        counts = torch.sum(count_mask.to(int), dim=-1)
        split_weights = torch.split(visit_PtL_weights.T, counts.tolist())
        visit_PtL_weights = torch.stack([torch.mean(group, dim=0) for group in split_weights])

        # Transformer
        transformer_mask = attn_mask.all(dim=-1)
        x = self.transformer_encoder(x, src_key_padding_mask=transformer_mask)

        # Retrive Last Visit
        visit_mask = attn_mask.all(dim=-1)
        inv_visit_mask = ~visit_mask
        last_visit_mask = inv_visit_mask.to(int)
        last_visit_mask = torch.sum(last_visit_mask, dim=-1) - 1
        x = x[torch.arange(B), last_visit_mask]

        pre_patient_PI_embeddings = x.clone()

        _, _, patient_prototypes, patient_PtL_weights = self.patient_PtL(x)
        x_patient = self.patient_PtP(x, patient_prototypes)
        x = x + x_patient
        patient_PtL_weights = patient_PtL_weights.T
        pre_PtF_embeddings = x.clone()

        # Prototype Fusion
        x_fused, PtF_weights, attn_weights = self.prototype_fusion(x, self.code_PtL.prototypes, self.visit_PtL.prototypes, self.patient_PtL.prototypes)
        x = x + x_fused
        post_PtF_embeddings = x.clone()
        out = self.head(x)

        prototypes = {'code_prototypes': self.code_PtL.prototypes, 'visit_prototypes': self.visit_PtL.prototypes, 'patient_prototypes': self.patient_PtL.prototypes}
        embeddings = {'original_code_embeddings': original_code_embeddings, 'pre_patient_PI_embeddings': pre_patient_PI_embeddings, 'pre_PtF_embeddings': pre_PtF_embeddings, 'post_PtF_embeddings': post_PtF_embeddings}
        weights = {'code_PtL_weights': code_PtL_weights, 'visit_PtL_weights': visit_PtL_weights, 'patient_PtL_weights': patient_PtL_weights, 'hierarchy_weights': PtF_weights, 'attn_weights': attn_weights}

        return {'logits': [out], 'prototypes': prototypes, 'embeddings': embeddings, 'weights': weights}


MODELS = {
    'ProtoEHR': ProtoEHR,
    'GraphCare': GraphCare,
    'KerPrint': KerPrint,
    'GRASP': GRASP,
    'StageNet': StageNet,
    'AdaCare': AdaCare,
    'Deepr': Deepr
}
