import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.func import jacfwd, vmap
from scripts.utils import getBCs
import torch.nn as nn

#This returns the approximation vector q(s) in R^2 associated to the boundary conditions (q1,q2,v1,v2)
#and correspondent to position s in the interval [0,1]

def eval_model(model,device,s,q1,q2,v1,v2):
    s_ = torch.tensor([[s]],dtype=torch.float32).to(device)
    q1 = torch.from_numpy(q1.astype(np.float32)).reshape(1,-1).to(device)
    q2 = torch.from_numpy(q2.astype(np.float32)).reshape(1,-1).to(device)
    v1 = torch.from_numpy(v1.astype(np.float32)).reshape(1,-1).to(device)
    v2 = torch.from_numpy(v2.astype(np.float32)).reshape(1,-1).to(device)
    return model(s_,q1,q2,v1,v2).detach().cpu().numpy()[0]

#This returns the approximation vector q'(s) in R^2 associated to the boundary conditions (q1,q2,v1,v2)
#and correspondent to position s in the interval [0,1]
# TODO : Understand why this is not providing a good fit of the derivative
def eval_derivative_model(model,device,s,q1,q2,v1,v2):
    s_ = torch.tensor([[s]],dtype=torch.float32).to(device)
    q1 = torch.from_numpy(q1.astype(np.float32)).reshape(1,-1).to(device)
    q2 = torch.from_numpy(q2.astype(np.float32)).reshape(1,-1).to(device)
    v1 = torch.from_numpy(v1.astype(np.float32)).reshape(1,-1).to(device)
    v2 = torch.from_numpy(v2.astype(np.float32)).reshape(1,-1).to(device)

    q = lambda s : model(s,q1,q2,v1,v2)[0]
    v = lambda s : (jacfwd(q))(s)[:,:,0].T
    return v(s_).detach().cpu().numpy().reshape(-1)


def plotTestResults(model,device,number_elements,number_components,trajectories):
    
    criterion = nn.MSELoss()
    
    bcs = getBCs(trajectories)
    q1 = bcs["q1"]
    q2 = bcs["q2"]
    v1 = bcs["v1"]
    v2 = bcs["v2"]
        
    xx = np.linspace(0, 1, number_elements+1)
    res = np.zeros((len(trajectories),2,len(xx)))
    res_derivative = np.zeros_like(res)

    for j in range(20):
        for i in range(len(xx)):
            res[j,:,i] = eval_model(model,device,xx[i],q1[j],q2[j],v1[j],v2[j])
            res_derivative[j,:,i] = eval_derivative_model(model,device,xx[i],q1[j],q2[j],v1[j],v2[j])
    
    fig = plt.figure(figsize=(10,5))
    for j in range(20):
        q_x_true = trajectories[j,np.arange(0,number_components,4)]
        q_y_true = trajectories[j,np.arange(1,number_components,4)]
        if j==0:
            plt.plot(q_x_true,q_y_true,'k-',label="True")
            plt.plot(res[j,0],res[j,1],'r--',label="Estimated")
            plt.legend()
        else:
            plt.plot(q_x_true,q_y_true,'k-')
            plt.plot(res[j,0],res[j,1],'r--')
            plt.legend()
        plt.xlabel(r"$q_1$")
        plt.ylabel(r"$q_2$")
        plt.title("All the trajectories")
    
    q_x_pred_torch = torch.from_numpy(res[:,0].astype(np.float32))
    q_x_true_torch = torch.from_numpy(trajectories[:,np.arange(0,number_components,4)].astype(np.float32))
    print("Mean squared error for q_x on all the set of trajectories : ",criterion(q_x_pred_torch,q_x_true_torch).item())
    
    q_y_pred_torch = torch.from_numpy(res[:,1].astype(np.float32))
    q_y_true_torch = torch.from_numpy(trajectories[:,np.arange(1,number_components,4)].astype(np.float32))
    print("Mean squared error for q_x on all the set of trajectories : ",criterion(q_x_pred_torch,q_x_true_torch).item())
    
    res = (q_x_pred_torch - q_x_true_torch)[:,0]
    zero = torch.zeros_like(res)
    print("Is the bc on q_x satisfied?",torch.allclose(res,zero))
    
    res = (q_y_pred_torch - q_y_true_torch)[:,0]
    zero = torch.zeros_like(res)
    print("Is the bc on q_y satisfied?",torch.allclose(res,zero))