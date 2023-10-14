import torch
import torch.nn as nn
from torch.func import jacfwd, vmap

class approximate_curve(nn.Module):
        def __init__(self,normalize=True,act_name='tanh', nlayers=3, hidden_nodes = 50, correct_functional=True, is_res=True, is_deeponet=False):
          super().__init__()

          if act_name=='tanh':
            self.act = lambda x : torch.tanh(x)
          elif act_name=="sin":
            self.act = lambda x : torch.sin(x)
          elif act_name=="swish":
            self.act = lambda x: x*torch.sigmoid(x)
          else:
            self.act = lambda x : torch.sigmoid(x)

          self.nlayers = nlayers
          self.is_norm = normalize
          self.is_res = is_res
          self.is_deeponet = is_deeponet
          self.hidden_nodes = hidden_nodes

          self.correct_functional = correct_functional

          self.lift = nn.Linear(9,self.hidden_nodes)
          ll = []
          for it in range(self.nlayers):
            ll.append(nn.Linear(self.hidden_nodes,self.hidden_nodes))
          self.linears = nn.ModuleList([ll[i] for i in range(self.nlayers)])

          # the next foru lines are layers of DeepONet trunk
          self.lift_U = nn.Linear(9,self.hidden_nodes)
          self.lift_V = nn.Linear(9,self.hidden_nodes)
          
          self.lift_H = nn.Linear(9,self.hidden_nodes) #added

          self.linears_Z = nn.ModuleList([nn.Linear(self.hidden_nodes,self.hidden_nodes) for i in range(self.nlayers)])
          
          self.proj = nn.Linear(self.hidden_nodes,2)


        #This is N in the markdown cell above
        def parametric_part(self,s,q1n,q2n,v1n,v2n):
                      
          s = s.reshape(-1,1)
          input = torch.cat((s,q1n,q2n,v1n,v2n),dim=1)
          
          if self.is_deeponet: 
              U = self.act(self.lift_U(input))
              V = self.act(self.lift_V(input))

              H = self.act(self.lift_H(input)) #added

              for i in range(self.nlayers):
                  Z = self.act(self.linears_Z[i](H))
                  H = U*(1-Z) + V*Z
            
              input = H

          else:
              
              input = self.act(self.lift(input))
    
              for i in range(self.nlayers):
                if self.is_res:
                    input = input + self.act(self.linears[i](input))
                else:
                    input = self.act(self.linears[i](input))
    
          output = self.proj(input)
          return output

        #find the coefficients a_0,a_1,a_2,a_3
        def get_coefficients(self,q1,q2,v1,v2):
          B = len(q1)
          q1n,q2n,v1n,v2n = self.normalize(q1,q2,v1,v2)
          left_node = torch.zeros((B,1),dtype=torch.float32).to(q1.device)
          right_node = torch.ones((B,1),dtype=torch.float32).to(q1.device)

          q = lambda s : self.parametric_part(s,q1n,q2n,v1n,v2n) #.reshape(-1,2)[0] tolto quando abbiamo riscritto gp_left &co
          
          g_left = q(left_node) 
          g_right = q(right_node)
          
          gp_left = (jacfwd(q))(left_node).sum(dim=(2,3)) # (jacfwd(q))(right_node)[:,:,0].T
          gp_right = (jacfwd(q))(right_node).sum(dim=(2,3)) # (jacfwd(q))(right_node)[:,:,0].T

          a0 = q1 - g_left
          a1 = v1 - gp_left
          a2 = 2*gp_left + gp_right - 3*g_right + 3*g_left - 3*q1 + 3*q2 - 2*v1 - v2
          a3 = -gp_right + 2*g_right - 2*g_left + 2*q1 - gp_left - 2*q2 + v1 + v2

          return a0,a1,a2,a3

        def correction_bcs(self,s,q1,q2,v1,v2):
          s = s.reshape(-1,1)
          a0,a1,a2,a3 = self.get_coefficients(q1,q2,v1,v2)
          a0,a1,a2,a3 = a0.to(q1.device),a1.to(q1.device),a2.to(q1.device),a3.to(q1.device)
          return a0 + a1*s + a2*s**2 + a3*s**3

        def normalize(self,q1,q2,v1,v2):
          q1n = (q1-1.5) / 1.5
          q2n = q2 #q2n = (q2+1) / 1 - 1
          v1n = v1 #(v1+3) / 3 - 1
          v2n = v2 #(v2+3) / 3 - 1
          return q1n,q2n,v1n,v2n

        #The forward method assembles q by combining the proper network "parametric_part"
        #with its computed correction which is polynomial in s.
        def forward(self,s,q1,q2,v1,v2):
          if self.is_norm:
            q1n,q2n,v1n,v2n = self.normalize(q1,q2,v1,v2)
            net_part = self.parametric_part(s,q1n,q2n,v1n,v2n)
          else:
            net_part = self.parametric_part(s,q1,q2,v1,v2)
          if self.correct_functional:
            correction_bcs = self.correction_bcs(s,q1,q2,v1,v2)
            return net_part + correction_bcs
          else:
            return net_part