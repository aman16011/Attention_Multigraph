# models.py
import torch
import torch.nn as nn
import torch.nn.functional as F
# from layers import GraphAttentionLayer, SpGraphAttentionLayer


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, dropout, alpha, nheads):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout
        self.nheads = nheads
        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
    
    def loss_function(self,output,target):
        target_D = target
        one_tensor = torch.tensor([1])
        one_tensor.expand(output.size())
        target_L = one_tensor-target
        loss_D = -1 * target_D * torch.log(torch.sigmoid(torch.mm(output,output.T)))
        loss_L = -1 * target_L * torch.log(1 - torch.sigmoid(torch.mm(output,output.T)))
        net_loss = loss_D + loss_L
        return torch.mean(torch.abs(net_loss))
      
    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        print(x.size())
        x = x.view(x.size()[0],self.nheads,-1)
        x = torch.mean(x,dim = 1)
        softmax = torch.nn.Softmax(dim=1)
        x = softmax(x,dim=0)
        return loss_function(x,adj)
