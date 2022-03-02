 '''First coding for simply two groups''' # Hard code()
        
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