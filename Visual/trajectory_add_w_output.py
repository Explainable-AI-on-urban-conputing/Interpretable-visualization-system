import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import pandas as pd
import time as ts
from ast import literal_eval
import numpy as np
from Guide_BP import *
import  cv2
import seaborn as sns
from params import *
from matplotlib import pyplot as plt
from IPython.core.pylabtools import figsize
import torch
import torch.nn as nn
def extrace_trajectory(timestamps,trajectory_line):
    #trajectory_line = pd.read_csv("./data/trajectory_line_0.csv", index_col=0).reset_index()
    #trajectory_line = pd.read_csv("./data/timeline", index_col=0).reset_index()
    if datasets_=='chengdu':
        W, H = 38, 36
    else:
        W, H = 39, 32
    train_data = np.zeros([144, 2,  W, H])
    trajectory_line=trajectory_line.iloc[timestamps-5:timestamps]
    trajectory_line.index = pd.Series([0, 1, 2, 3, 4])
    # train data
    n_0 = len(literal_eval(trajectory_line.iloc[0]['trajectory_inflow']))
    n_1 = len(literal_eval(trajectory_line.iloc[1]['trajectory_inflow']))
    n_2 = len(literal_eval(trajectory_line.iloc[2]['trajectory_inflow']))
    n_3 = len(literal_eval(trajectory_line.iloc[3]['trajectory_inflow']))
    n_4 = len(literal_eval(trajectory_line.iloc[4]['trajectory_inflow']))
    X_0, X_1, X_2, X_3, X_4 = np.zeros([n_0, 2,  W, H]), np.zeros([n_1, 2,  W, H]), np.zeros(
        [n_2, 2,  W, H]), np.zeros([n_3, 2,  W, H]),np.zeros([n_4, 2,  W, H])
    train_list=[]
    X=[]
    ranges=0
    for name, i in trajectory_line.iterrows():
        trajectory_inflow = literal_eval(i['trajectory_inflow'])
        for j in range(len(trajectory_inflow)):
            ranges=np.max([list(trajectory_inflow[j].keys())[0],ranges])
    train_list.append([X_0,X_1,X_2,X_3,X_4])
    list_0 = [[] for i in range(ranges+1)]
    list_1 = [[] for i in range(ranges+1)]
    traj_list,gps_list=[],[]
    for v in range(5):
        traj_list.append([[] for x in range(eval('n_'+str(v)))])
        gps_list.append([[] for x in range(eval('n_' + str(v)))])
    for name, i in trajectory_line.iterrows():
        train_data=train_list[0][name]
        trajectory_inflow=literal_eval(i['trajectory_inflow'])
        trajectory_outflow=literal_eval(i['trajectory_outflow'])
        gps_in = literal_eval(i['trajectory_gps_in'])
        for j in range(len(trajectory_inflow)):
            _ = [trajectory_inflow[j][key] for key in trajectory_inflow[j]][0]
            tmp,s=[],0
            for i, item in gps_in[j].items():
                tmp+=item
            gps_list[name][j].append(tmp)
            for k in _:
                if k!=-1:
                    x=int(k/H)
                    y=k%H
                    train_data[j,0,x,y]+=1
                    traj_list[name][j].append([x,y])

            if len(gps_list[name][j][0])!=0:
                #print(gps_list[name][j][0])
                list_0[list(trajectory_inflow[j].keys())[0]].append({name: j})
                list_1[list(trajectory_outflow[j].keys())[0]].append({name: j})
            #print(gps_list[name][j])
            _ = [trajectory_outflow[j][key] for key in trajectory_outflow[j]][0]
            for k in _:
                if k!=-1:
                    x=int(k/H)
                    y=k%H
                    train_data[j,1,x,y]+=1
        X.append(train_data)
        a, b,trajectory_list = [], [],[list_0,list_1]
        for i in range(len(trajectory_list[0])):
            if len(trajectory_list[0][i]) != 0:
                a.append(trajectory_list[0][i])
            if len(trajectory_list[1][i]) != 0:
                b.append(trajectory_list[1][i])
    return X[0],X[1],X[2],X[3],X[4],[a,b],traj_list,gps_list

def recreate_grad_image(im_as_var,channel):

    #recreated_im = im_as_var.cpu().numpy()
    recreated_im = im_as_var
    reverse_mean = np.mean(recreated_im,axis=(1,2))
    reverse_std =  np.std(recreated_im,axis=(1,2))
    for c in range(channel):
        recreated_im[c] /= reverse_std[c]
        recreated_im[c] -= reverse_mean[c]
    recreated_im[recreated_im > 1] = 1
    recreated_im[recreated_im < 0] = 0
    recreated_im = np.round(recreated_im * 255)
    recreated_im = np.uint8(recreated_im)
    return recreated_im
