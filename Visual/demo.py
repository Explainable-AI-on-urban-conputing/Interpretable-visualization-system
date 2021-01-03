from flask import Flask,render_template,request,redirect,url_for
app = Flask(__name__)
import numpy as np
from trajectory_add_w_output import *
import json
import shap
@app.route('/form', methods=['POST', 'GET'])
def bio_data_form():
    if request.method == "POST":
        username = request.form['username']
        age = request.form['age']
        email = request.form['email']
        hobbies = request.form['hobbies']
        return redirect(url_for('showbio',
                                username=username,
                                age=age,
                                email=email,
                                hobbies=hobbies))
    return render_template("bio_form.html")

def compute_traj_grad(x,y):
    timestamps = 48
    model = torch.load('./result/best_model.pkl').inputs_c.eval()
    path = "./data/Chengdu/timeline1"
    trajectory_line = pd.read_csv(path, index_col=0).reset_index()
    X_0, X_1, X_2, X_3, X_4, trajectory_list, traj_list, gps_list = extrace_trajectory(timestamps, trajectory_line)
    x_0 = torch.from_numpy(X_0).type(torch.FloatTensor).cuda()
    x_1 = torch.from_numpy(X_1).type(torch.FloatTensor).cuda()
    x_2 = torch.from_numpy(X_2).type(torch.FloatTensor).cuda()
    x_3 = torch.from_numpy(X_3).type(torch.FloatTensor).cuda()
    x_4 = torch.from_numpy(X_4).type(torch.FloatTensor).cuda()
    channel_0, channel_1, channel_2, channel_3, channel_4 = torch.sum(x_0, dim=0), torch.sum(x_1, dim=0), \
    torch.sum(x_2, dim=0), torch.sum(x_3, dim=0), torch.sum(x_4, dim=0)
    test_c = torch.cat([channel_0, channel_1, channel_2, channel_3, channel_4], dim=0).unsqueeze(0) / MAX_FLOWIO
    test_c = nn.Parameter(test_c)
    optimizer = torch.optim.Adam([test_c])
    # guide_model=GuidedBackpropReLUModel(model)
    X = [X_0, X_1, X_2, X_3, X_4]
    out=model(test_c)
    optimizer.zero_grad()
    out[0,0,x,y].backward()
    grad=test_c.grad[0,0::2].cpu().detach().numpy()
    traj_grad_=[]
    for traj in trajectory_list[0]:
        traj_grad=0
        for v in traj:
            for i,item in v.items():
                traj_grad += np.sum(X[i][item, 0] * grad[i])
        traj_grad_.append(traj_grad)
    keys=[str(x) for x in np.arange(len(traj_grad_))]
    traj_grad_=dict(zip(keys,traj_grad_))
    return traj_grad_


@app.route('/showbio', methods=['GET'])
def showbio():
    index = int(request.args.get('index'))
    x = int(index/36)
    y = index%36
    traj_grad_=compute_traj_grad(x,y)
    return render_template("show_bio.html",
                           index=index)


if __name__ == '__main__':
    app.run(debug = True,port=8000)