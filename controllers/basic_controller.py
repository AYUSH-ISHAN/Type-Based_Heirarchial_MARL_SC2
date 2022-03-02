from modules.agents import REGISTRY as agent_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY
import torch as th
from modules.agents.thgc_agent import THGCAgent

group = 2

# This multi-agent controller shares parameters between agents
class BasicMAC:
    def __init__(self, scheme, groups, args):
        self.n_agents = args.n_agents
        self.args = args
        input_shape = self._get_input_shape(scheme)
        # print(input_shape)
        self.thgc = THGCAgent(input_shape, self.args)
        self._build_agents(input_shape, self.args)
        self.agent_output_type = args.agent_output_type
        self.grouping_method = args.method1 # for location based grouping or args.method2 for health based grouping
        self.action_selector = action_REGISTRY[args.action_selector](args)

        self.hidden_states = None

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        # Only select actions for the selected batch elements in bs
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        agent_outputs = self.forward(ep_batch, t_ep, test_mode=test_mode)
        chosen_actions = self.action_selector.select_action(agent_outputs[bs], avail_actions[bs], t_env, test_mode=test_mode)
        return chosen_actions
    '''
    for t in range(batch.max_seq_length):
            agent_outs = self.mac.forward(batch, t=t)
            mac_out.append(agent_outs)
    '''
    # look at 't' above
    def forward(self, ep_batch, t, test_mode=False):
        batch_input = self._build_inputs(ep_batch, t)
        agents_groups = self._build_grps_wise_inp(batch_input, ep_batch[self.grouping_method][:, t, ...], ep_batch.device)
        adj_mat = ep_batch["adj_matrix"][:, t, ...]
        avail_actions = ep_batch["avail_actions"][:, t]

        agent_wise_input = list(th.unbind(agents_groups), dim=0)

        for agent_group in agents_groups:
            for agent in agent_group:
                agent_inputs = agent_wise_input[agent]
                h,self.hidden_states = self.agent(agent_inputs, self.hidden_states, adj_mat,intra=True, first=True)
            list1 = []
            '''Check whether below stacking is needed or not.'''
            for k in h:
                list1.append(k)
            hg = th.stack(list1, dim=0) # stacking all agents inputs
            for agent in agent_group:
                agent_inputs = agent_wise_input[agent]
                z,self.hidden_states = self.agent(hg, self.hidden_states, adj_mat,intra=True,GAT_intra=True) # after mlp layer
            h_,self.hidden_states = self.agent(h, self.hidden_states, adj_mat,intra=True) # after mlp layer
            H = th.stack((h_, z),dim=0) # stacking along row
            V = th.sum(H,dim=1)  # do aggregate sum taking along a row (representing a group)
            Vg = th.stack((k for k in V), dim=0)
            Qg,self.hidden_states = self.agent(Vg, self.hidden_states, adj_mat,inter=True,GAT_intra=True)
            Ql,self.hidden_states = self.agent(V, self.hidden_states, adj_mat,inter=True)

        agent_outs = th.stack((Ql, Qg),dim=0)#, self.hidden_states = self.agent(agent_inputs, self.hidden_states)

        # Softmax the agent outputs if they're policy logits
        if self.agent_output_type == "pi_logits":

            if getattr(self.args, "mask_before_softmax", True):
                # Make the logits for unavailable actions very negative to minimise their affect on the softmax
                reshaped_avail_actions = avail_actions.reshape(ep_batch.batch_size * self.n_agents, -1)
                agent_outs[reshaped_avail_actions == 0] = -1e10

            agent_outs = th.nn.functional.softmax(agent_outs, dim=-1)
            if not test_mode:
                # Epsilon floor
                epsilon_action_num = agent_outs.size(-1)
                if getattr(self.args, "mask_before_softmax", True):
                    # With probability epsilon, we will pick an available action uniformly
                    epsilon_action_num = reshaped_avail_actions.sum(dim=1, keepdim=True).float()

                agent_outs = ((1 - self.action_selector.epsilon) * agent_outs
                               + th.ones_like(agent_outs) * self.action_selector.epsilon/epsilon_action_num)

                if getattr(self.args, "mask_before_softmax", True):
                    # Zero out the unavailable actions
                    agent_outs[reshaped_avail_actions == 0] = 0.0

        return agent_outs.view(ep_batch.batch_size, self.n_agents, -1)

    def init_hidden(self, batch_size):
        self.hidden_states = self.agent.init_hidden().unsqueeze(0).expand(batch_size, self.n_agents, -1)  # bav

    def parameters(self):
        return list(self.agent.parameters())+self.gat_params()
    def load_state(self, other_mac):
        self.agent.load_state_dict(other_mac.agent.state_dict())
        self.load_gat_state(other_mac)
        

    def cuda(self):
        self.agent.cuda()
        self.gat_cuda()
        
    def save_models(self, path):
        th.save(self.agent.state_dict(), "{}/agent.th".format(path))
        self.save_gat(path)

    def load_models(self, path):
        self.agent.load_state_dict(th.load("{}/agent.th".format(path), map_location=lambda storage, loc: storage))
        self.gat_load(path)

    def _build_agents(self, input_shape):  
        self.agent = agent_REGISTRY[self.args.agent](input_shape, self.args)
        

    def _build_inputs(self, batch, t):
        # Assumes homogenous agents with flat observations.
        # Other MACs might want to e.g. delegate building inputs to each agent
        bs = batch.batch_size
        inputs = []
        inputs.append(batch["obs"][:, t])  # b1av
        if self.args.obs_last_action:
            if t == 0:
                inputs.append(th.zeros_like(batch["actions_onehot"][:, t]))
            else:
                inputs.append(batch["actions_onehot"][:, t-1])
        if self.args.obs_agent_id:
            inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1))

        # inputs = th.cat([x.reshape(bs*self.n_agents, -1) for x in inputs], dim=1)
        inputs = th.cat([x.reshape(bs*self.n_agents, -1) for x in inputs], dim=1)
        return inputs

    def _get_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"]
        if self.args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0]
        if self.args.obs_agent_id:
            input_shape += self.n_agents

        return input_shape

    
    def _build_grps_wise_inp(self, batch, batch_size, groups, device):
        # batch_size is 32
        inp_obs = batch.reshape(batch_size, self.n_agents, -1).to(device=device)
        agent_inputs = []
        for group in groups:
            for agent in group:
                inps = [inp_obs[:,k,:] for k in agent]
                agent_inputs.append(inps)
        return th.tensor(agent_inputs)

        
