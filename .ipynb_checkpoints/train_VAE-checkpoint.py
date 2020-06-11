import torch
import numpy as np
import torch.nn.functional as F
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, BatchNorm1d as BN
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader
from torch_geometric.nn import PointConv, fps, radius, global_max_pool
from chamfer_distance import ChamferDistance
from Data import ShapeNet_2048
from train_AE import SAModule, GlobalSAModule, MLP
from sklearn.model_selection import train_test_split

BETA = 1e-6
Bottle = 128
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.sa1_module = SAModule(0.5, 0.2, MLP([3 + 3, 64, 64, 128]))
        self.sa2_module = SAModule(0.25, 0.4, MLP([128 + 3, 128, 128, 256]))
        self.sa3_module = GlobalSAModule(MLP([256 + 3, 256, 256, 1024])) 
        

        self.mu_lin = Lin(1024,Bottle)
        self.sig_lin = Lin(1024,Bottle)
        
        self.lin1 = Lin(Bottle, 1024)
        self.lin2 = Lin(1024, 2048)
        self.lin3 = Lin(2048, 2048 * 3)
    
    
    def encode(self, x, batch):
        sa1_out = self.sa1_module(x, x, batch)
        sa2_out = self.sa2_module(*sa1_out)
        sa3_out = self.sa3_module(*sa2_out)
        x, pos, batch = sa3_out
        
        return self.mu_lin(x) , self.sig_lin(x)
        

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        z = F.relu(self.lin1(z))
        z = F.relu(self.lin2(z))
        z = self.lin3(z)
        return z
    
    def forward(self, data):
        x,  batch = data.x, data.batch
        mu, logvar = self.encode(x, batch)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


def train(epoch):
    model.train()
    total_loss = 0
    step = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        dist1, dist2 = criterion(out[0].reshape(-1,2048,3), data.x.reshape(-1,2048,3))
        CHM = (torch.mean(dist1)) + (torch.mean(dist2)) 
        KLD = BETA * -0.5 * torch.mean(
            torch.sum(1 + out[2] - out[1].pow(2) - out[2].exp(),dim=1))
        loss = CHM + KLD
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
        if step % 10 == 0:
            print(data)
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tLoss_recons: {:.6f}\tLoss_kld: {:.6f}'.format(
                epoch, step * len(data), len(train_loader.dataset),
                100. * step / len(train_loader),
                loss.item() / len(data), 
                CHM.item()/ len(data),
                KLD.item()/len(data)))
        step += 1
    return total_loss / len(train_dataset)



if __name__ == '__main__':

    path = '../data/shapenet_2048'
    dataset = ShapeNet_2048(path, split='trainval', categories='Chair')
    indices = np.arange(len(dataset))
    train_indeces, _ = train_test_split(indices, test_size=.25, random_state=42)
    train_dataset = dataset[torch.tensor(train_indeces)]
    print(len(train_dataset))
    train_loader = DataLoader(
        train_dataset, batch_size=32, shuffle=True)
    device = torch.device('cuda')
    model = Net().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    print(model)
    print('Training started:')
    criterion = ChamferDistance()
    for epoch in range(1, 401):
        loss = train(epoch)
        print('Epoch {:03d}, Loss: {:.4f}'.format(
            epoch, loss))
        if epoch % 20 ==0:
            torch.save(model.state_dict(),'./saved_models/pointVAE_{}_{}_Ch'.format(Bottle,BETA)+'{}'.format(epoch)+'.pt')