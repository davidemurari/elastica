from torch.func import jacfwd,vmap
import torch
import torch.nn as nn
import numpy as np

def trainModel(number_elements,device,model,criterion,optimizer,epochs,trainloader,train_with_tangents=False,pde_regularisation=False):
  lossVal = 1.
  epoch = 1

  scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=45, gamma=0.1)

  x_eval = np.linspace(0,1,number_elements+1)

  gamma_tangents = 1e-2
  gamma_phys = 1e-3

  count = 0
  cc = 0
  while epoch < epochs:
      losses = []
      running_loss = 0

      for i, inp in enumerate(trainloader):
          q1,q2,v1,v2,s,_,qs,vs = inp
          q1,q2,v1,v2,s,qs,vs = q1.to(device),q2.to(device),v1.to(device),v2.to(device),s.to(device),qs.to(device),vs.to(device)

          def closure():
              optimizer.zero_grad()

              #Training with only positions
              res_q = model(s,q1,q2,v1,v2)
              loss = criterion(res_q,qs)

              if train_with_tangents and epoch>70:
                q = lambda s : model(s.reshape(-1,1),q1,q2,v1,v2)[0]
                v = lambda s : jacfwd(q)(s)
                res_v = (jacfwd(q))(s)[:,:,0].T.reshape(-1,2)
                loss += gamma_tangents * criterion(res_v,vs)

              #Not tested this part
              if pde_regularisation and epoch>50:
                q = lambda s : model(s.reshape(-1,1),q1,q2,v1,v2)
                v = lambda s : jacfwd(q)(s)
                a = lambda s: jacfwd(v)(s)

                lhs = lambda s: torch.linalg.norm(a(s),dim=1)**2# + (torch.linalg.norm(q(s),dim=1)**2-1)**2  #write the correct expression of the PDE, which should be lhs(s)=0
                lhs_evaluated = (lhs)(s)
                rhs = torch.zeros_like(lhs_evaluated)
                loss += gamma_phys * criterion(lhs_evaluated,rhs)

              loss.backward()

              return loss

          optimizer.step(closure)

      model.eval();
      with torch.no_grad():
        res_q = model(s,q1,q2,v1,v2)
        loss = criterion(res_q,qs)
        print(f'Loss [{epoch+1}](epoch): ', loss.item())
        if torch.isnan(loss):
          epoch = epochs + 10
          break
      model.train();

      epoch += 1

      scheduler.step()

  print('Training Done')
  return loss