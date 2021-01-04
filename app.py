from flask import Flask,render_template,request,redirect,url_for
import json
import params
import os
from io import StringIO
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from trajectory_add_w_output import *
app = Flask(__name__)



@app.route('/show',methods=['GET'])
def show():
    index = request.args.get('index')
    return render_template("show.html",index=index)

@app.route('/',methods=['POST','GET'])
def visual():
    if request.method == "POST":
            index = request.form['index']
            index=int(index)
            x = int(index/36)
            y = index % 36
            traj_grad_ = compute_traj_grad(x, y)
            return redirect(url_for('show',
                                    index=index))
    return render_template("visual_pro.html")

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
# @app.route("/")
# def index():
#   return render_template("index.html")

# @app.route("/cool_form", methods=["GET", "POST"])
# def cool_form():
#   if request.method == "POST":
#     # do stuff when the form is submitted
#
#     # redirect to end the POST handling
#     # the redirect can be to the same route or somewhere else
#     return redirect(url_for("index"))
#
#   # show the form, it wasn"t submitted
#   return render_template("cool_form.html")




          
          
if __name__ == '__main__':
    app.run(FLASK_DEBUG=True,port=8080)
