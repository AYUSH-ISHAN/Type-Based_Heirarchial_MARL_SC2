import torch
import torch.nn as nn
import torch.nn.functional as F
from GAT.GAT import GAT
import enum

'''
Here, all the changes will be done regarding the agents.


Check the hyperparmaters

'''
CORA_NUM_INPUT_FEATURES = 1433
CORA_NUM_CLASSES = 7

# Support for 3 different GAT implementations 

'''Try out three and see which one works for you'''
class LayerType(enum.Enum):
    IMP1 = 0,
    IMP2 = 1,
    IMP3 = 2

gat_config = {
        "num_of_layers": 2,  # GNNs, contrary to CNNs, are often shallow (it ultimately depends on the graph properties)
        "num_heads_per_layer": [8, 1],
        "num_features_per_layer": [CORA_NUM_INPUT_FEATURES, 8, CORA_NUM_CLASSES],
        "add_skip_connection": False,  # hurts perf on Cora
        "bias": True,  # result is not so sensitive to bias
        "dropout": 0.6,  # result is sensitive to dropout
        "layer_type": LayerType.IMP3  # fastest implementation enabled by default
    }


class THGCAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(THGCAgent, self).__init__()
        self.args = args
        self.input_shape = input_shape
        self.hidden_dim_2 = None    #   Adust the dim here
        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        #self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)  # here GAT + LSTM networks will go
        self.gat = GAT(args.rnn_hidden_dim, gat_config["num_heads_per_layer"], gat_config["num_features_per_layer"])
        self.lstm = nn.LSTMCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        ##  Now concat there result and then feed to fc2..  In the forward function.
        
        '''GAT implementaion is there.. But for LSTM use nn.LSTM'''
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)


    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    # this hidden_state is set to None in the argument itself. OR it is set by init_hidden as -
    '''        self.hidden_states = self.agent.init_hidden().unsqueeze(0).expand(batch_size, self.n_agents, -1)  # bav
    '''
    
    def MLP_layer(self, inputs, layer, activation=False):
        if layer==1:
            fcn_dim1, fcn_dim2 = self.input_shape, self.args.rnn_hidden_dim
        if layer == 2:
            fcn_dim1, fcn_dim2 = self.args.rnn_hidden_dim, self.hidden_dim_2
        if layer == 3:
            fcn_dim1, fcn_dim2 = self.hidden_dim_2, self.args.n_actions

        fcn = nn.Linear(fcn_dim1, fcn_dim2)
        if activation:
            x = F.relu(fcn(inputs))
        else:
            x = F.relu(fcn(inputs))
        
        return x
    


    def forward(self, inputs, hidden_state, num_groups):

        n_agents = self.args['n_agents']
        '''Here, num_groups denotes number of groups of agents we are making'''
        # for agent_group in range(num_groups):
        #     for agent_id in range(agent_group*(int(n_agents/num_groups)), agent_group*(int(n_agents/num_groups)) + int(n_agents/num_groups)):
        #         x = F.relu(self.fc1(inputs))  # customise the inputs
        #         h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        #         #h = self.rnn(x, h_in)
        #         h = torch.concat([self.gat(x, h_in), self.lstm(h_in)]) # torch.concat((self.gat, self.lstm))
        #         q = self.fc2(h)
        #         h_agent_group.append(h)
        #         q_agent_group.append(q)
                
                
        # h, q = torch.tensor(h), torch.tensor(q)  # I think there is no need for outputs 
        # return q, h
        '''First coding for simply two groups'''
        x11 = F.relu(self.fc1(inputs))  # customise the inputs
        x12 = F.relu(self.fc1(inputs))  # customise the inputs
        x13 = F.relu(self.fc1(inputs))  # customise the inputs
        x14 = F.relu(self.fc1(inputs))  # customise the inputs
        x21 = F.relu(self.fc1(inputs))  # customise the inputs
        x22 = F.relu(self.fc1(inputs))  # customise the inputs
        x23 = F.relu(self.fc1(inputs))  # customise the inputs
        x24 = F.relu(self.fc1(inputs))  # customise the inputs

        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)

        g11 = self.gat(torch.concat([x11, x12, x13, x14]), h_in)
        g12 = self.gat(torch.concat([x11, x12, x13, x14]), h_in)
        g13 = self.gat(torch.concat([x11, x12, x13, x14]), h_in)
        g14 = self.gat(torch.concat([x11, x12, x13, x14]), h_in)
        g21 = self.gat(torch.concat([x21, x22, x23, x24]), h_in)
        g22 = self.gat(torch.concat([x21, x22, x23, x24]), h_in)
        g23 = self.gat(torch.concat([x21, x22, x23, x24]), h_in)
        g24 = self.gat(torch.concat([x21, x22, x23, x24]), h_in)
        
        l11 = self.lstm(x11)
        l12 = self.lstm(x12)
        l13 = self.lstm(x13)
        l14 = self.lstm(x14)
        l21 = self.lstm(x21)
        l22 = self.lstm(x22)
        l23 = self.lstm(x23)
        l24 = self.lstm(x24)

        lo11 = self.fc2(l11)
        lo12 = self.fc2(l12)
        lo13 = self.fc2(l13)
        lo14 = self.fc2(l14)
        lo21 = self.fc2(l21)
        lo22 = self.fc2(l22)
        lo23 = self.fc2(l23)
        lo24 = self.fc2(l24)

        go11 = self.fc2(g11)
        go12 = self.fc2(g12)
        go13 = self.fc2(g13)
        go14 = self.fc2(g14)
        go21 = self.fc2(g21)
        go22 = self.fc2(g22)
        go23 = self.fc2(g23)
        go24 = self.fc2(g24)

        V1 = torch.concat([lo11,lo12,lo13,lo14,go11,go12,go13,go14])
        V2 = torch.concat([lo21,lo22,lo23,lo24,go21,go22,go23,go24])

        V1_g = self.gat(torch.concat([V1, V2]))
        V2_g = self.gat(torch.concat([V1, V2]))

        Q1 = torch.concat([self.fc3(V1), self.fc3(V1_g)])
        Q2 = torch.concat([self.fc3(V2), self.fc3(V2_g)])
        '''Now fedding Q1 and Q2 in the mixed networks'''


# def _build_inputs(self, batch, t):
#         # Assumes homogenous agents with flat observations.
#         # Other MACs might want to e.g. delegate building inputs to each agent
#         bs = batch.batch_size
#         inputs = []
#         inputs.append(batch["obs"][:, t])  # b1av
#         if self.args.obs_last_action:
#             if t == 0:
#                 inputs.append(th.zeros_like(batch["actions_onehot"][:, t]))
#             else:
#                 inputs.append(batch["actions_onehot"][:, t-1])
#         if self.args.obs_agent_id:
#             inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1))

#         inputs = th.cat([x.reshape(bs*self.n_agents, -1) for x in inputs], dim=1)
#         return inputs



# def _get_input_shape(self, scheme):
#         input_shape = scheme["obs"]["vshape"]
#         if self.args.obs_last_action:
#             input_shape += scheme["actions_onehot"]["vshape"][0]
#         if self.args.obs_agent_id:
#             input_shape += self.n_agents

#         return input_shape