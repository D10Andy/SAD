import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_scatter as scatter
from modules.utils import MergeLayer_output, Feat_Process_Layer, drop_edge
from modules.embedding_module import get_embedding_module
from modules.time_encoding import TimeEncode
from model.gdn import graph_deviation_network
from model.supconloss import SupConLoss


class TGAT(torch.nn.Module):
    def __init__(self, config, device):
        super().__init__()
        self.cfg = config

        self.nodes_dim = self.cfg.input_dim
        self.edge_dim = self.cfg.input_dim
        self.dims = self.cfg.hidden_dim

        self.n_heads = self.cfg.n_heads
        self.dropout = self.cfg.drop_out
        self.n_layers = self.cfg.n_layer

        self.mode = self.cfg.mode

        self.time_encoder = TimeEncode(dimension=self.dims)
        self.embedding_module_type = self.cfg.module_type
        self.embedding_module = get_embedding_module(module_type=self.embedding_module_type,
                                                     time_encoder=self.time_encoder,
                                                     n_layers=self.n_layers,
                                                     node_features_dims=self.dims,
                                                     edge_features_dims=self.dims,
                                                     time_features_dim=self.dims,
                                                     hidden_dim=self.dims,
                                                     n_heads=self.n_heads, dropout=self.dropout)

        self.node_preocess_fn = Feat_Process_Layer(self.nodes_dim, self.dims)
        self.edge_preocess_fn = Feat_Process_Layer(self.edge_dim, self.dims)


        self.affinity_score = MergeLayer_output(self.dims, self.dims, drop_out=0.2)


        self.gdn = graph_deviation_network(self.cfg, device)
        self.suploss = SupConLoss()


    def forward(self, src_org_edge_feat, src_edge_to_time, src_center_node_idx, src_neigh_edge, src_node_features,
                current_time, label):
        # apply tgat
        source_node_embedding = self.compute_temporal_embeddings(src_neigh_edge, src_edge_to_time,
                                                                                src_org_edge_feat, src_node_features)

        root_embedding = source_node_embedding[src_center_node_idx, :]
        anom_score = self.gdn(root_embedding, current_time, label)  # 异常检测
        dev, group = self.gdn.dev_diff(torch.squeeze(anom_score), current_time, label)

        # 节点分类
        logits = self.affinity_score(root_embedding)

        # 样本增强
        aug_neigh_edge, aug_edge_to_time, aug_org_edge_feat = drop_edge(src_neigh_edge,src_edge_to_time,src_org_edge_feat,
                                                                        0.2)
        aug_node_embedding = self.compute_temporal_embeddings(aug_neigh_edge, aug_edge_to_time,
                                                                                aug_org_edge_feat, src_node_features)
        aug_node_embedding = aug_node_embedding[src_center_node_idx, :]

        prediction_dict = {}
        prediction_dict['logits'] = torch.reshape(logits, [-1])
        prediction_dict['anom_score'] = anom_score
        prediction_dict['time'] = current_time
        prediction_dict['root_embedding'] = torch.cat([root_embedding, aug_node_embedding], dim=0)
        prediction_dict['group'] = group.clone().detach().repeat(2, 1)
        prediction_dict['dev'] = dev.clone().detach().repeat(2, 1)
        
        return prediction_dict


    def compute_temporal_embeddings(self, neigh_edge, edge_to_time, edge_feat, node_feat):
        node_feat = self.node_preocess_fn(node_feat)
        edge_feat = self.edge_preocess_fn(edge_feat)

        node_embedding = self.embedding_module.compute_embedding(neigh_edge, edge_to_time,
                                                                 edge_feat, node_feat)

        return node_embedding
