import argparse
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from B_code.utils import load_data, load_graph, adj_knn1, normalize_adj, numpy_to_torch
from B_code.GNN_previous import GNNLayer
from B_code.eva_previous import eva
from datetime import datetime
from sklearn.cluster import KMeans
from torch.nn import Linear
from torch.nn.parameter import Parameter
from torch.optim import Adam

tic = time.time()
TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())

# Define the DAE module
class DAE(nn.Module):
    def __init__(self, n_enc_1, n_enc_2, n_enc_3, n_dec_1, n_dec_2, n_dec_3,
                 n_input, n_z):
        super(DAE, self).__init__()
        # Encoder
        self.enc_1 = Linear(n_input, n_enc_1)
        self.enc_2 = Linear(n_enc_1, n_enc_2)
        self.enc_3 = Linear(n_enc_2, n_enc_3)
        # extracted feature by DAE
        self.z_layer = Linear(n_enc_3, n_z)
        # Decoder
        self.dec_1 = Linear(n_z, n_dec_1)
        self.dec_2 = Linear(n_dec_1, n_dec_2)
        self.dec_3 = Linear(n_dec_2, n_dec_3)
        self.x_bar_layer = Linear(n_dec_3, n_input)
    def forward(self, x):
        enc_z2 = F.relu(self.enc_1(x))
        enc_z3 = F.relu(self.enc_2(enc_z2))
        enc_z4 = F.relu(self.enc_3(enc_z3))
        z = self.z_layer(enc_z4)
        dec_z2 = F.relu(self.dec_1(z))
        dec_z3 = F.relu(self.dec_2(dec_z2))
        dec_z4 = F.relu(self.dec_3(dec_z3))
        x_bar = self.x_bar_layer(dec_z4)
        return x_bar, enc_z2, enc_z3, enc_z4, z

class MLP_L(nn.Module):
    def __init__(self, n_mlp):
        super(MLP_L, self).__init__()
        self.wl = Linear(n_mlp, 5)
    def forward(self, mlp_in):
        weight_output = F.softmax(F.leaky_relu(self.wl(mlp_in)), dim=1)
        return weight_output
class MLP_1(nn.Module):
    def __init__(self, n_mlp):
        super(MLP_1, self).__init__()
        self.w1 = Linear(n_mlp,2)
    def forward(self, mlp_in):
        weight_output = F.softmax(F.leaky_relu(self.w1(mlp_in)), dim=1) 
        return weight_output
class MLP_2(nn.Module):
    def __init__(self, n_mlp):
        super(MLP_2, self).__init__()
        self.w2 = Linear(n_mlp, 2)
    def forward(self, mlp_in):
        weight_output = F.softmax(F.leaky_relu(self.w2(mlp_in)), dim=1)
        return weight_output
class MLP_3(nn.Module):
    def __init__(self, n_mlp):
        super(MLP_3, self).__init__()
        self.w3 = Linear(n_mlp, 2)
    def forward(self, mlp_in):
        weight_output = F.softmax(F.leaky_relu(self.w3(mlp_in)), dim=1)  
        return weight_output
class MLP_AG(nn.Module):
    def __init__(self, n_mlp):
        super(MLP_AG, self).__init__()
        self.w = Linear(n_mlp,2)
    def forward(self, mlp_in):
        weight_output1 = F.softmax(F.leaky_relu(self.w(mlp_in)), dim=1) 
        return weight_output1

# Define the GACN module
class GACN(nn.Module):

    def __init__(self, n_enc_1, n_enc_2, n_enc_3, n_dec_1, n_dec_2, n_dec_3, 
                n_input, n_z, n_clusters, n_x, v=1):
        super(GACN, self).__init__()

        self.ae = DAE(
            n_enc_1=n_enc_1,
            n_enc_2=n_enc_2,
            n_enc_3=n_enc_3,
            n_dec_1=n_dec_1,
            n_dec_2=n_dec_2,
            n_dec_3=n_dec_3,
            n_input=n_input,
            n_z=n_z)

        if args.name == 'amap' or args.name == 'pubmed':
            pretrained_dict = torch.load(args.pretrain_path, map_location='cpu')
            model_dict = self.ae.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.ae.load_state_dict(model_dict)
        else:
            self.ae.load_state_dict(torch.load(args.pretrain_path, map_location='cpu'))

        # Define the GEL module
        self.gacn_0 = GNNLayer(n_input, n_enc_1)
        self.gacn_1 = GNNLayer(n_enc_1, n_enc_2)
        self.gacn_2 = GNNLayer(n_enc_2, n_enc_3)
        self.gacn_3 = GNNLayer(n_enc_3, n_z)
        self.gacn_z = GNNLayer(3020, n_clusters)
        self.mlp1    = MLP_1(2*n_enc_1)
        self.mlp2    = MLP_2(2*n_enc_2)
        self.mlp3    = MLP_3(2*n_enc_3)
        self.agcn_ag = MLP_AG(2*n_x)
        self.mlp  = MLP_L(3020)
        self.cluster_layer = Parameter(torch.Tensor(n_clusters, n_z))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)
        self.v = v

    def forward(self, x, x_z, adj, iter_t, knn_k):
        #
        # DAE
        #
        x_bar, h1, h2, h3, z = self.ae(x)
        x_array = list(np.shape(x))
        n_x = x_array[0]

        if x.equal(x_z):
            adj = adj
        else:
            # V1
            adj1_tensor = adj
            z4adj2 = x_z.data.cpu().numpy()
            adj2 = adj_knn1(z4adj2)
            adj2 = adj2.cuda()
            adj2_tensor = adj2
            if adj1_tensor.is_sparse:
                adj1_tensor = adj1_tensor.to_dense()
            if adj2_tensor.is_sparse:
                adj2_tensor = adj2_tensor.to_dense()
            wg = self.agcn_ag( torch.cat((adj1_tensor,adj2_tensor), 1) )
            wg = F.normalize(wg,p=2)
            wg1 = torch.reshape(wg[:,0], [n_x, 1])
            wg2 = torch.reshape(wg[:,1], [n_x, 1])
            wg1_broadcast = wg1.repeat(1,n_x)
            wg2_broadcast = wg2.repeat(1,n_x)
            adj = wg1_broadcast.mul(adj1_tensor) + wg2_broadcast.mul(adj2_tensor)

        # Feature Representation Learning
        z1 = self.gacn_0(x, adj)
        m1 = self.mlp1( torch.cat((h1,z1), 1) )
        m1 = F.normalize(m1,p=2)
        m11 = torch.reshape(m1[:,0], [n_x, 1])
        m12 = torch.reshape(m1[:,1], [n_x, 1])
        m11_broadcast =  m11.repeat(1,500)
        m12_broadcast =  m12.repeat(1,500)
        z2 = self.gacn_1( m11_broadcast.mul(z1)+m12_broadcast.mul(h1), adj)
        m2 = self.mlp2( torch.cat((h2,z2),1) )     
        m2 = F.normalize(m2,p=2)
        m21 = torch.reshape(m2[:,0], [n_x, 1])
        m22 = torch.reshape(m2[:,1], [n_x, 1])
        m21_broadcast = m21.repeat(1,500)
        m22_broadcast = m22.repeat(1,500)
        z3 = self.gacn_2( m21_broadcast.mul(z2)+m22_broadcast.mul(h2), adj)
        m3 = self.mlp3( torch.cat((h3,z3),1) )   
        m3 = F.normalize(m3,p=2)
        m31 = torch.reshape(m3[:,0], [n_x, 1])
        m32 = torch.reshape(m3[:,1], [n_x, 1])
        m31_broadcast = m31.repeat(1,2000)
        m32_broadcast = m32.repeat(1,2000)
        z4 = self.gacn_3( m31_broadcast.mul(z3)+m32_broadcast.mul(h3), adj)

        u  = self.mlp(torch.cat((z1,z2,z3,z4,z),1))
        u  = F.normalize(u,p=2) 
        u0 = torch.reshape(u[:,0], [n_x, 1])
        u1 = torch.reshape(u[:,1], [n_x, 1])
        u2 = torch.reshape(u[:,2], [n_x, 1])
        u3 = torch.reshape(u[:,3], [n_x, 1])
        u4 = torch.reshape(u[:,4], [n_x, 1])
        tile_u0 = u0.repeat(1,500)
        tile_u1 = u1.repeat(1,500)
        tile_u2 = u2.repeat(1,2000)
        tile_u3 = u3.repeat(1,10)
        tile_u4 = u4.repeat(1,10)
        net_output = torch.cat((tile_u0.mul(z1), tile_u1.mul(z2), tile_u2.mul(z3), tile_u3.mul(z4), tile_u4.mul(z)), 1 )   
        net_output = self.gacn_z(net_output, adj, active=False) 
        predict = F.softmax(net_output, dim=1)
        q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
        q = q.pow((self.v + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()

        return x_bar, q, predict, z, net_output, adj

# P, the auxiliary distribution
def p_distribution(q):
    weight = q**2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()

# The main function
def train_GACN(dataset):

    dataname = 'dblp'
    eprm_state = 'EGRCNet'

    # Record the experimental results
    file_out = open('./R_output/'+dataname+'_'+eprm_state+'.out', 'a')
    print("The experimental results", file=file_out)

    # The trade-off parameters, where the better result can be obtained by fine-tuning the ones
    lambda_1 = [0.01] # [0.001, 0.01, 0.1, 1, 10, 100, 1000] # [0.001] # 
    lambda_2 = [0.1] # [0.001, 0.01, 0.1, 1, 10, 100, 1000] # [10] # 
    lambda_3 = [0.1] # [0.001, 0.01, 0.1, 1, 10, 100, 1000] # [10] # 

    for ld1 in lambda_1:
            for ld2 in lambda_2:
                for ld3 in lambda_3:

                    print("lambda_1: ", ld1,  "lambda_2: ", ld2, "lambda_3: ", ld3, file=file_out)

                    # raw data
                    data = torch.Tensor(dataset.x).cuda()
                    y = dataset.y

                    # model parameters
                    model = GACN(500, 500, 2000, 2000, 500, 500,
                                n_input=args.n_input,
                                n_z=args.n_z,
                                n_clusters=args.n_clusters,
                                n_x = args.n_x,
                                v=1.0).cuda()
                    optimizer = Adam(model.parameters(), lr=args.lr)

                    # KNN Graph
                    if args.name == 'amap' or args.name == 'pubmed':
                        load_path = "data/" + args.name + "/" + args.name
                        adj = np.load(load_path+"_adj.npy", allow_pickle=True)
                        adj = normalize_adj(adj, self_loop=True, symmetry=False)
                        adj = numpy_to_torch(adj, sparse=True).to(torch.device("cuda")) # opt.args.device = torch.device("cuda" if opt.args.cuda else "cpu")
                    else:
                        adj = load_graph(args.name, args.k)
                        adj = adj.cuda()

                    # KNN Graph
                    if args.name == 'amap' or args.name == 'pubmed':
                        load_path = "data/" + args.name + "/" + args.name
                        adj = np.load(load_path+"_adj.npy", allow_pickle=True)
                        adj = normalize_adj(adj, self_loop=True, symmetry=False)
                        adj = numpy_to_torch(adj, sparse=True).to(torch.device("cuda")) # opt.args.device = torch.device("cuda" if opt.args.cuda else "cpu")
                    else:
                        adj = load_graph(args.name, args.k)
                        adj = adj.cuda()

                    with torch.no_grad():
                        _, _, _, _, z = model.ae(data)

                    iters10_ACC_iter_Z = []
                    iters10_NMI_iter_Z = []
                    iters10_ARI_iter_Z = []
                    iters10_F1_iter_Z  = []

                    z_1st = z
                    iter_t = 1

                    kmeans = KMeans(n_clusters=args.n_clusters, n_init=20)
                    y_pred = kmeans.fit_predict(z_1st.data.cpu().numpy())
                    model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).cuda()
                    acc,nmi,ari,f1 = eva(y, y_pred, 'pae')

                    ACC_iter_Z = []
                    NMI_iter_Z = []
                    ARI_iter_Z = []
                    F1_iter_Z  = []

                    loss = 0

                    for epoch in range(200):
                        
                        _, tmp_q, pred, _, net_output, net_adj = model(data, data, adj, iter_t, 1)

                        tmp_q = tmp_q.data
                        p = p_distribution(pred.data)

                        # Z
                        resZ = pred.data.cpu().numpy().argmax(1)
                        acc,nmi,ari, f1 = eva(y, resZ, str(epoch) + 'Z')
                        ACC_iter_Z.append(acc)
                        NMI_iter_Z.append(nmi)
                        ARI_iter_Z.append(ari)
                        F1_iter_Z.append(f1)

                        if epoch % 10 == 0 and epoch > 0:
                            x_bar, q, pred, _, net_output, net_adj = model(data, pred.data, adj, iter_t, 1)
                            iter_t = iter_t + 1
                        else:
                            x_bar, q, pred, _, net_output, net_adj = model(data, data, adj, iter_t, 1)

                        re_loss     = F.mse_loss(x_bar, data)
                        
                        kl_qp_loss  = F.kl_div(q.log(), p, reduction='batchmean')
                        kl_pq_loss  = F.kl_div(p.log(), q, reduction='batchmean')
                        
                        kl_zp_loss  = F.kl_div(pred.log(), p, reduction='batchmean')
                        kl_pz_loss  = F.kl_div(p.log(), pred, reduction='batchmean')

                        kl_qz_loss  = F.kl_div(q.log(), pred, reduction='batchmean')
                        kl_zq_loss  = F.kl_div(pred.log(), q, reduction='batchmean')

                        # Eq. (12): {Reconstruction loss}+{Jeffreys divergence loss}
                        loss = \
                            re_loss \
                            + ld1 * (kl_qp_loss + kl_pq_loss) \
                                + ld2 * (kl_qz_loss + kl_zq_loss) \
                                    + ld3 * ( kl_zp_loss + kl_pz_loss) \

                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                    # Clustering assignments
                    acc_max= np.max(ACC_iter_Z)
                    nmi_max= np.max(NMI_iter_Z)
                    ari_max= np.max(ARI_iter_Z)
                    f1_max= np.max(F1_iter_Z)
                    iters10_ACC_iter_Z.append(round(acc_max,5))
                    iters10_NMI_iter_Z.append(round(nmi_max,5))
                    iters10_ARI_iter_Z.append(round(ari_max,5))
                    iters10_F1_iter_Z.append(round(f1_max,5))
                    print("################ iters10_ACC_iter_Z   #####################", file=file_out)
                    print("Z_ACC mean",round(np.mean(iters10_ACC_iter_Z),5),"max",np.max(iters10_ACC_iter_Z),"\n",iters10_ACC_iter_Z, file=file_out)
                    print("Z_NMI mean",round(np.mean(iters10_NMI_iter_Z),5),"max",np.max(iters10_NMI_iter_Z),"\n",iters10_NMI_iter_Z, file=file_out)
                    print("Z_ARI mean",round(np.mean(iters10_ARI_iter_Z),5),"max",np.max(iters10_ARI_iter_Z),"\n",iters10_ARI_iter_Z, file=file_out)
                    print("Z_F1 mean",round(np.mean(iters10_F1_iter_Z),5),"max",np.max(iters10_F1_iter_Z),"\n",iters10_F1_iter_Z, file=file_out)
                    print('Z_:acc, nmi, ari, f1: \n{:.4f}\n{:.4f}\n{:.4f}\n{:.4f}'.format(round(np.mean(iters10_ACC_iter_Z),5),round(np.mean(iters10_NMI_iter_Z),5),round(np.mean(iters10_ARI_iter_Z),5),round(np.mean(iters10_F1_iter_Z),5)), file=file_out)
                    
                    print('Z_:acc, nmi, ari, f1: \n{:.4f}\n{:.4f}\n{:.4f}\n{:.4f}'.format(round(np.mean(iters10_ACC_iter_Z),5),round(np.mean(iters10_NMI_iter_Z),5),round(np.mean(iters10_ARI_iter_Z),5),round(np.mean(iters10_F1_iter_Z),5)))

    file_out.close()

if __name__ == "__main__":

    for i in range(1):
        parser = argparse.ArgumentParser(
            description='train',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser.add_argument('--name', type=str, default='dblp')
        parser.add_argument('--k', type=int, default=1)
        parser.add_argument('--lr', type=float, default=1e-3)
        parser.add_argument('--n_clusters', default=3, type=int)
        parser.add_argument('--n_z', default=10, type=int)
        parser.add_argument('--pretrain_path', type=str, default='pkl')
        args = parser.parse_args()
        args.cuda = torch.cuda.is_available()
        print("use cuda: {}".format(args.cuda))
        device = torch.device("cuda" if args.cuda else "cpu")

        args.pretrain_path = 'data/{}.pkl'.format(args.name)
        dataset = load_data(args.name)

        if args.name == 'usps':
            args.k = 1
            args.n_clusters = 10
            args.n_input = 256
            args.n_x = 9298

        if args.name == 'hhar':
            args.k = 1
            args.n_clusters = 6
            args.n_input = 561
            args.n_x = 10299

        if args.name == 'reut':
            args.k = 1
            args.lr = 1e-4
            args.n_clusters = 4
            args.n_input = 2000
            args.n_x = 10000

        if args.name == 'acm':
            args.k = 1
            args.n_clusters = 3
            args.n_input = 1870
            args.n_x = 3025

        if args.name == 'dblp':
            args.k = 1
            args.n_clusters = 4
            args.n_input = 334
            args.n_x = 4057

        if args.name == 'cite':
            args.lr = 1e-4
            args.k = 1
            args.n_clusters = 6
            args.n_input = 3703
            args.n_x = 3327

        if args.name == 'pubmed':
            args.k = 1
            args.n_clusters = 3
            args.n_input = 500
            args.n_x = 19717

        if args.name == "STL10":
            args.lr = 1e-3
            args.k = 3
            args.n_clusters = 10
            args.n_input = 512
            args.n_x = 13000

        if args.name == "img10":
            args.lr = 1e-3
            args.k = 3
            args.n_clusters = 10
            args.n_input = 512
            args.n_x = 13000

        print(args)
        train_GACN(dataset)

    # Running time
    toc = time.time()
    print("Time:", (toc - tic))