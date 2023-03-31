import torch
from manifolds import Hyperboloid, ManifoldParameter
from layers import LorentzLayer
import numpy as np


class HGCF(torch.nn.Module):
    def __init__(
        self,
        num_users: int,
        num_items: int,
        graph,
        c: int = 1,
        num_layers=4,
        embedding_dim=64,
        margin=1.0,
        weight_decay=1e-5,
        scale=0.1,
        network="resSumGCN"
        ):
        super().__init__()
        
        self.c = torch.tensor([c]).cuda()
        self.manifold = Hyperboloid()
        self.graph = graph.cuda()

        self.num_users, self.num_items = num_users, num_items
        self.num_nodes = num_users + num_items
        self.margin = margin
        self.weight_decay = weight_decay
        self.num_layers = num_layers
        
        self.embedding = torch.nn.Embedding(num_embeddings=self.num_nodes, embedding_dim=embedding_dim).cuda()

        self.embedding.state_dict()['weight'].uniform_(-scale, scale)
        self.embedding.weight = torch.nn.Parameter(self.manifold.expmap0(self.embedding.state_dict()['weight'], self.c))

        self.embedding.weight = ManifoldParameter(self.embedding.weight, True, self.manifold, self.c)

        hgc_layers = []
        hgc_layers.append(LorentzLayer(self.manifold, embedding_dim, self.c, network, num_layers))
        self.convs = torch.nn.Sequential(*hgc_layers)
        
        self.encode_graph = True
        
    def compute_embedding(self):
        """
        Get embedding of all nodes.
        :return: [num_nodes, embedding_dim]
        """
        x = self.embedding.weight
        x_hyp = self.manifold.proj(x, c=self.c)
        
        if self.encode_graph is True:
            out, _ = self.convs((x_hyp, self.graph))
        else:
            out = self.convs(x_hyp)

        return out
    
    def forward(self, edge_index):
        """
        Forward propagation.
        :param edge_index: [2, num_edges]
        :return: [num_edges, 1]
        """
        out = self.compute_embedding()
        
        out_src = out[edge_index[0]]
        out_dst = out[edge_index[1]]
        
        sqdist = self.manifold.sqdist(out_src, out_dst, self.c)
        
        return sqdist
        
    def margin_loss(self, edge_index, neg_edge_index):
        """
        Compute loss for given edge index.
        :param edge_index: [2, num_edges]
        :param neg_edge_index: [2, num_edges]
        :return: loss
        """
        pos_scores = self.forward(edge_index)
        neg_scores = self.forward(neg_edge_index)

        loss = pos_scores - neg_scores + self.margin
        loss[loss < 0] = 0
        loss = torch.sum(loss)
        
        return loss

    def get_user_rating(self):
        """
        Get rating of user for all items.
        :param user_id: user id
        :return: [num_items]
        """
        out = self.compute_embedding()
        probs_matrix = np.zeros((self.num_users, self.num_items))
        
        for user in range(self.num_users):
            user_embedding = out[user]
            user_embedding = user_embedding.repeat(self.num_items).view(self.num_items, -1)
            item_embeddings = out[np.arange(self.num_users, self.num_nodes), :]
            sqdist = self.manifold.sqdist(user_embedding, item_embeddings, self.c)

            probs = sqdist.detach().cpu().numpy() * -1
            probs_matrix[user] = np.reshape(probs, [-1, ])
            
        return probs_matrix
