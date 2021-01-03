import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from torch.autograd import Function
import torch
import  numpy as np
import torch.nn as nn
from ast import literal_eval
from trajectory_add_w_output import *
class GuidedBackpropReLU(Function):

    @staticmethod
    def forward(self, input):
        positive_mask = (input > 0).type_as(input)
        output = torch.addcmul(torch.zeros(input.size()).type_as(input), input, positive_mask)
        self.save_for_backward(input, output)
        return output

    @staticmethod
    def backward(self, grad_output):
        input, output = self.saved_tensors
        grad_input = None

        positive_mask_1 = (input > 0).type_as(grad_output)
        positive_mask_2 = (grad_output > 0).type_as(grad_output)
        grad_input = torch.addcmul(torch.zeros(input.size()).type_as(input),
                                   torch.addcmul(torch.zeros(input.size()).type_as(input), grad_output,
                                                 positive_mask_1), positive_mask_2)

        return grad_input


class GuidedBackpropReLUModel:
    def __init__(self, model):
        self.model = model
        self.model.eval()
        def recursive_relu_apply(module_top):
            for idx, module in module_top._modules.items():
                recursive_relu_apply(module)
                if module.__class__.__name__ == 'ReLU':
                    module_top._modules[idx] = GuidedBackpropReLU.apply

        # replace ReLU with GuidedBackpropReLU
        recursive_relu_apply(self.model)

    def forward(self, c):
        return self.model(c)

    def __call__(self, c):
        output = self.forward(c)

        return output

def Intergrated_grad(input,num_steps):
    baseline=torch.zeros(input.shape).cuda()
    interpolated_input=torch.zeros([num_steps+1]+list(input.shape[1:])).to('cuda')
    for step in range(num_steps + 1):
        interpolated_input[step]= baseline + (step / num_steps) * (input - baseline)
    return interpolated_input

def Smooth_add_nois(input,num_steps,sigam):

    interpolated_input=torch.zeros([num_steps+1]+list(input.shape[1:])).to('cuda')
    for step in range(num_steps + 1):
        noise = input.data.new(input.size()).normal_(0, sigam ** 2).cuda()
        t=input + noise/MAX_FLOWIO
        t[t<0]=0
        interpolated_input[step]= t
    return interpolated_input
def Ablation(input,num_steps,x,y):
    baseline=torch.zeros(input.shape).cuda()
    interpolated_input=input.repeat(num_steps+1,1,1,1)
    for step in range(num_steps + 1):
        interpolated_input[step,:,x,y]= baseline[0,:,x,y]  + (step / num_steps) * (input[0,:,x,y] - baseline[0,:,x,y])
    #print(interpolated_input[:,0,x,y])
    return interpolated_input
def split_dict(trajectory_gps_in,trajectory_gps_out):
    inflow,outflow=[],[]
    for name, i in trajectory_gps_in.iteritems():
        inflow.append(literal_eval(i))
    for name, i in trajectory_gps_out.iteritems():
        outflow.append(literal_eval(i))
    return inflow,outflow


def grad(outputs,val_c,optimizer,model,start):
    optimizer.zero_grad()
    end=start+outputs.shape[0]
    out=torch.sum(outputs[:,0,5,3])
    out.backward(retain_graph=True)
    grad=torch.sum(val_c.grad[start:end,1:2:5],1)
    #print(torch.max(grad))
    grad=torch.max(grad, torch.ones_like(grad) - 1)
    grad=grad.flatten(start_dim=1)
    grad=softmax(grad)
    label=softmax(od_label[start:end])
    loss=torch.sum(label*torch.log(label/grad))
    return loss

def dy_dx_loss(outputs,Train_c,model,start):
    out=torch.sum(outputs[:,0,5,3])
    out.backward()
    grad=Train_c.grad()
    grad[grad<0]=0
    loss=torch.sum(grad)
    return loss

weight=[]
def print_TrainableEltwiseLayer(module_top):
    for idx, module in module_top._modules.items():
        print_TrainableEltwiseLayer(module)
        if module.__class__.__name__ == 'TrainableEltwiseLayer':
            weight.append(module.weights.cpu().detach().numpy())
    return  weight

weight=[]
def print_linear(module_top):
    for idx, module in module_top._modules.items():
        print_TrainableEltwiseLayer(module)
        if module.__class__.__name__ == 'linear':
            weight.append(module.weights.cpu().detach().numpy())
    return  weight

def add_trajectory_noise(timestamps,X_0, X_1, X_2, X_3, X_4, trajectory_list):
    X_0 = torch.from_numpy(X_0).type(torch.FloatTensor).cuda()
    X_1 = torch.from_numpy(X_1).type(torch.FloatTensor).cuda()
    X_2 = torch.from_numpy(X_2).type(torch.FloatTensor).cuda()
    X_3 = torch.from_numpy(X_3).type(torch.FloatTensor).cuda()
    X_4 = torch.from_numpy(X_4).type(torch.FloatTensor).cuda()
    x_0, x_1, x_2, x_3, x_4 = torch.zeros(X_0.shape).cuda(), torch.zeros(X_1.shape).cuda(), \
                              torch.zeros(X_2.shape).cuda(), torch.zeros(X_3.shape).cuda(), torch.zeros(X_4.shape).cuda()
    for i in range(len(trajectory_list[0])):
        if len(trajectory_list[0][i]) != 0:
            vars()['W_' + str(i)] = torch.normal(mean=torch.ones(1,2),std=torch.ones(1,2)*1).unsqueeze(2).unsqueeze(2).cuda()
            for j in range(len(trajectory_list[0][i])):
                keys = list(trajectory_list[0][i][j].keys())[0]
                values = list(trajectory_list[0][i][j].values())[0]
                eval('x_' + str(keys))[values] = eval('W_' + str(i)) * eval('X_' + str(keys))[values]
    channel_0, channel_1, channel_2, channel_3, channel_4 = torch.sum(x_0, dim=0), torch.sum(x_1, dim=0), \
                                                            torch.sum(x_2, dim=0), torch.sum(x_3, dim=0), torch.sum(x_4,dim=0)
    test_c = torch.cat([channel_0, channel_1, channel_2, channel_3, channel_4], dim=0).unsqueeze(0)
    return test_c/MAX_FLOWIO

