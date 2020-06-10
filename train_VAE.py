import torch
import torch.nn.functional as F
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, BatchNorm1d as BN
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader
from torch_geometric.nn import PointConv, fps, radius, global_max_pool
from chamfer_distance import ChamferDistance
from Data import ShapeNet_2048
from train_AE import SAModule, GlobalSAModule, MLP


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.sa1_module = SAModule(0.2, 0.2, MLP([3 + 3, 64, 64, 128]))
        self.sa2_module = SAModule(0.25, 0.4, MLP([128 + 3, 128, 128, 256]))
        self.sa3_module = GlobalSAModule(MLP([256 + 3, 256, 256, 512])) 
        
        self.mu_lin = Lin(512,20)
        self.sig_lin = Lin(512,20)
        
        self.lin1 = Lin(20, 512)
        self.lin2 = Lin(512, 1024)
        self.lin3 = Lin(1024, 2048 * 3)
    
    
    
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


def train():
    model.train()
    total_loss = 0
    step = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        dist1, dist2 = criterion(out[0].reshape(-1,2048,3), data.x.reshape(-1,2048,3))
        CHM = (torch.mean(dist1)) + (torch.mean(dist2)) 
        KLD = -0.5 * torch.mean(
            torch.sum(1 + out[2] - out[1].pow(2) - out[2].exp(),dim=1))
        loss = CHM + KLD
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
        if step % 50 == 0:
            print(step)
            print('KLD:',KLD.item()/len(data))
            print('CHM:',CHM.item()/len(data))
        step += 1
    return total_loss / len(dataset)



if __name__ == '__main__':

    path = '../data/shapenet_2048'
    dataset = ShapeNet_2048(path, split='trainval', categories='Chair')
    print(dataset[0])
    train_loader = DataLoader(
        dataset, batch_size=32, shuffle=True)
    device = torch.device('cuda')
    model = Net().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    print(model)
    print('Training started:')
    criterion = ChamferDistance()
    for epoch in range(1, 401):
        loss = train()
        print('Epoch {:03d}, Loss: {:.4f}'.format(
            epoch, loss))
        if epoch % 10 ==0:
            torch.save(model.state_dict(),'./pointVAECh'+'{}'.format(epoch)+'.pt')