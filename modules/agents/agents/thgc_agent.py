import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
from torch_geometric.utils import dense_to_sparse

class THGCAgent(nn.Module):
    def __init__(self, input_shape, args):

        # inputs = th.cat([x.reshape(bs*self.n_agents, -1) for x in inputs], dim=1)
        # look at input shape here

        super(THGCAgent, self).__init__()
        self.args = args
        self.input_shape = 'look above'#input_shape
        
        self.hidden_dim_2 = None    #   Adust the dim here
        self.fc1 = nn.Linear(input_shape, args.mlp_hidden_dim)
        # self.gat = GAT(args.mlp_hidden_dim*5, gat_config["num_heads_per_layer"], gat_config["num_features_per_layer"])
        self.lstm = nn.LSTMCell(args.mlp_hidden_dim, args.mlp_hidden_dim)
        
        self.fc2 = nn.Linear(args.mlp_hidden_dim, args.n_actions)


    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.mlp_hidden_dim).zero_()

    # this hidden_state is set to None in the argument itself. OR it is set by init_hidden as -
    ''' self.hidden_states = self.agent.init_hidden().unsqueeze(0).expand(batch_size, self.n_agents, -1)  # bav
    '''
    
    def MLP_layer(self, inputs, layer, activation=False):
        if layer==1:
            fcn_dim1, fcn_dim2 = self.input_shape, self.args.mlp_hidden_dim
        if layer == 2:
            fcn_dim1, fcn_dim2 = self.args.mlp_hidden_dim, self.hidden_dim_2
        if layer == 3:
            fcn_dim1, fcn_dim2 = self.hidden_dim_2, self.args.n_actions

        fcn = nn.Linear(fcn_dim1, fcn_dim2)
        if activation:
            x = F.relu(fcn(inputs))
        else:
            x = fcn(inputs)
        
        return x

    def GAT_layer(self, inputs, adj, layer):

        if layer==1:
            gat_dim1, gat_dim2 = self.input_shape, self.args.hid_msg_dim
            adj = None
        if layer == 2:
            gat_dim1, gat_dim2 = self.args.hid_msg_dim, self.args.msg_out_dim
            adj = None

        self.gat = GATComm(gat_dim1, gat_dim2)
        x = self.gat(inputs, adj)
        
        return x
    
    def forward(self, inputs, hidden_state, adj_mat, intra=False, inter=False, GAT_intra=False, GAT_inter=False, first=False):

        n_agents = self.args['n_agents']

        if intra:
            if first:
                x1 = self.MLP_layer(self, inputs, 1, activation=True)
                return x1
            if not GAT_intra:
                x2h = self.lstm(inputs)
                x2 = self.MLP_layer(self, x2h, 2, activation=True)
            else:
                x2h = self.GAT_layer(inputs, adj_mat, layer=1)  # combined x2 input of aself.gents in a particular group
                x2 = self.MLP_layer(self, x2h, 2, activation=True)

            return x2, x2h 

        if inter:
            # in this case inputs will be V
            if GAT_inter:
                x3g = self.GAT_layer(inputs, adj_mat, layer=2)  # combined x3 input of different groups
                Q = self.MLP_layer(self, x3g, 3, activation=True)
            else:
                Q = self.MLP_layer(self, inputs, 3, activation=True)
        
        return Q, x3g


    def load_state(self, other_mac):  
        self.gat.load_state_dict(other_mac.gat.state_dict())
    
    def gat_params(self):
        return list(self.gat.parameters())
    
    def load_gat_state(self, other_mac):
        self.gat.load_state_dict(other_mac.gat.state_dict())

    def gat_cuda(self):
        self.gat.cuda_transfer()

    def save_gat(self, path):
        torch.save(self.gat.state_dict(), "{}/gat.th".format(path))

    def gat_load(self, path):
        self.gat.load_state_dict(torch.load("{}/gat.th".format(path), map_location=lambda storage, loc: storage))

class GATComm(torch.nn.Module):
    def __init__(self, input_shape, hidden_msg_dim, hid_msg_out,args, training=True):
        super(GATComm, self).__init__()
        self.args = args

        self.convs = []
        self.convs.append(GCNConv(input_shape, hidden_msg_dim))
        for i in range(1,self.args.num_layers-1):
            self.convs.append(GCNConv(hidden_msg_dim, hidden_msg_dim))
        self.convs.append(GCNConv(hidden_msg_dim, hid_msg_out))    
    
    def forward(self,x, adj_matrix):
        x_out = []
        for x_in, am_in in zip(torch.unbind(x, dim=0), torch.unbind(adj_matrix, dim=0)):
            for i in range(self.args.num_layers):
                x_in = self.convs[i](x_in, dense_to_sparse(am_in)[0])
                if (i+1)<self.args.num_layers:
                    x_in = F.elu(x_in)
                    x_in = F.dropout(x_in, p=0.2, training=self.training)
            x_out.append(x_in) 
        return torch.stack(x_out, dim=0)
       
