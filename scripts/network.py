import torch
import torch.nn as nn
from torch.func import jacfwd

class approximate_curve(nn.Module):
        def __init__(self,normalize=True,act_name='tanh', nlayers=3, hidden_nodes = 50, correct_functional=True):
          super().__init__()

          if act_name=='tanh':
            self.act = lambda x : torch.tanh(x)
          elif act_name=="sin":
            self.act = lambda x : torch.sin(x)
          elif act_name=="relu2":
            self.act = lambda x: torch.relu(x)**2
          else:
            self.act = lambda x : torch.sigmoid(x)

          self.nlayers = nlayers
          self.is_norm = normalize
          self.hidden_nodes = hidden_nodes

          self.correct_functional = correct_functional

          self.lift = nn.Linear(9,self.hidden_nodes)
          ll = []
          for it in range(self.nlayers):
            ll.append(nn.Linear(self.hidden_nodes,self.hidden_nodes))
          self.linears = nn.ModuleList([ll[i] for i in range(self.nlayers)])

          self.proj = nn.Linear(self.hidden_nodes,2)


        #This is N in the markdown cell above
        def parametric_part(self,s,q1n,q2n,v1n,v2n):

          input = torch.cat((s,q1n,q2n,v1n,v2n),dim=1)

          input = self.act(self.lift(input))

          for i in range(self.nlayers):
            input = input + self.act(self.linears[i](input))

          output = self.proj(input)


          return output

        #find the coefficients a_0,a_1,a_2,a_3
        def get_coefficients(self,q1,q2,v1,v2):
          B = len(q1)
          q1n,q2n,v1n,v2n = self.normalize(q1,q2,v1,v2)
          left_node = torch.zeros((B,1),dtype=torch.float32).to(q1.device)
          right_node = torch.ones((B,1),dtype=torch.float32).to(q1.device)

          q = lambda s : self.parametric_part(s,q1n,q2n,v1n,v2n).reshape(-1,2)[0]

          g_left = self.parametric_part(left_node,q1n,q2n,v1n,v2n)
          g_right = self.parametric_part(right_node,q1n,q2n,v1n,v2n)

          gp_left = (jacfwd(q))(left_node)[:,:,0].T
          gp_right = (jacfwd(q))(right_node)[:,:,0].T

          a0 = q1 - g_left
          a1 = v1 - gp_left
          a2 = 2*gp_left + gp_right + -3*g_right + 3*g_left - 3*q1 + 3*q2 - 2*v1-v2
          a3 = -gp_right + 2*g_right - 2*g_left + 2*q1 - gp_left - 2*q2 + v1 + v2

          return a0,a1,a2,a3

        def correction_bcs(self,s,q1,q2,v1,v2):

          B = len(q1)

          left_node = torch.zeros((B,1),dtype=torch.float32).to(q1.device)
          right_node = torch.ones((B,1),dtype=torch.float32).to(q1.device)

          a0,a1,a2,a3 = self.get_coefficients(q1,q2,v1,v2)
          a0,a1,a2,a3 = a0.to(q1.device),a1.to(q1.device),a2.to(q1.device),a3.to(q1.device)
          return a0 + a1*s + a2*s**2 + a3*s**3

        def normalize(self,q1,q2,v1,v2):
          q1n = (q1+3) / 3 - 1
          q2n = (q2+3) / 3 - 1
          v1n = (v1+3) / 3 - 1
          v2n = (v2+3) / 3 - 1
          return q1n,q2n,v1n,v2n

        #The forward method assembles q by combining the proper network "parametric_part"
        #with its computed correction which is polynomial in s.
        def forward(self,s,q1,q2,v1,v2):
          if self.is_norm:
            q1,q2,v1,v2 = self.normalize(q1,q2,v1,v2)
          net_part = self.parametric_part(s,q1,q2,v1,v2)
          if self.correct_functional:
            correction_bcs = self.correction_bcs(s,q1,q2,v1,v2)
            return net_part + correction_bcs
          else:
            return net_part