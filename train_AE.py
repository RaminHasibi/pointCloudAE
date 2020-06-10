import torch
import torch.nn.functional as F
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, BatchNorm1d as BN
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader
from torch_geometric.nn import PointConv, fps, radius, global_max_pool
from chamfer_distance import ChamferDistance
from Data import ShapeNet_2048


class SAModule(torch.nn.Module):
    def __init__(self, ratio, r, nn):
        super(SAModule, self).__init__()
        self.ratio = ratio
        self.r = r
        self.conv = PointConv(nn)

    def forward(self, x, pos, batch, num_samples=32):
        idx = fps(pos, batch, ratio=self.ratio)
        row, col = radius(pos, pos[idx], self.r, batch, batch[idx],
                          max_num_neighbors=num_samples)
        edge_index = torch.stack([col, row], dim=0)
        x = self.conv(x, (pos, pos[idx]), edge_index)
        pos, batch = pos[idx], batch[idx]
        return x, pos, batch


class GlobalSAModule(torch.nn.Module):
    def __init__(self, nn):
        super(GlobalSAModule, self).__init__()
        self.nn = nn

    def forward(self, x, pos, batch):
        x = self.nn(torch.cat([x, pos], dim=1))
        x = global_max_pool(x, batch)
        pos = pos.new_zeros((x.size(0), 3))
        batch = torch.arange(x.size(0), device=batch.device)
        return x, pos, batch


def MLP(channels, batch_norm=True):
    return Seq(*[
        Seq(Lin(channels[i - 1], channels[i]), ReLU(), BN(channels[i]))
        for i in range(1, len(channels))
    ])

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.sa1_module = SAModule(0.2, 0.2, MLP([3 + 3, 64, 64, 128]))
        self.sa2_module = SAModule(0.25, 0.4, MLP([128 + 3, 128, 128, 256]))
        self.sa3_module = GlobalSAModule(MLP([256 + 3, 256, 512, 1024])) 
        
        self.lin1 = Lin(1024, 1024)
        self.lin2 = Lin(1024, 2048)
        self.lin3 = Lin(2048, 2048 * 3)

    def forward(self, data):
        sa1_out = self.sa1_module(data.x, data.x, data.batch)
        sa2_out = self.sa2_module(*sa1_out)
        sa3_out = self.sa3_module(*sa2_out)
        
        x, pos, batch = sa3_out

        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = self.lin3(x)
        return x
    



def train():
    model.train()
    total_loss = 0
    step = 0
    for data in train_loader:
        step += 1
        if step % 50 == 0:
            print(step)
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        dist1, dist2 = criterion(out.reshape(-1,2048,3), data.x.reshape(-1,2048,3))
        loss = (torch.mean(dist1)) + (torch.mean(dist2)) 
        loss.backward()
        total_loss += loss.item() * data.num_graphs
        optimizer.step()
    return total_loss / len(dataset)



if __name__ == '__main__':

    path = '../../data/shapenet_2048'
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
            torch.save(model.state_dict(),'./pointAECh'+'{}'.format(epoch)+'.pt')