import math
import os

import torch
from loguru import logger
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import RGCNConv

class KGencoder(nn.Module):
    def __init__(self, num_relations, n_entity, edge_index, edge_type):
        super(KGencoder, self).__init__()
        self.hidden_size = 128
        self.num_bases = None
        self.kg_encoder = RGCNConv(in_channels=self.hidden_size, out_channels=self.hidden_size, num_relations=num_relations, num_bases=self.num_bases)
        
        
        self.node_embeds = nn.Parameter(torch.empty(n_entity, self.hidden_size))
        stdv = math.sqrt(6.0 / (self.node_embeds.size(-2) + self.node_embeds.size(-1)))
        self.node_embeds.data.uniform_(-stdv, stdv)

        self.edge_index = nn.Parameter(edge_index, requires_grad=False)
        self.edge_type = nn.Parameter(edge_type, requires_grad=False)


    def get_entity_embeds(self):
        node_embeds = self.node_embeds
        entity_embeds = self.kg_encoder(node_embeds, self.edge_index, self.edge_type) + node_embeds
        return entity_embeds


    def forward(self):
        entity_embeds = self.get_entity_embeds()
        return entity_embeds



