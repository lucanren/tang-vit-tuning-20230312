import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch import nn
from torch import Tensor
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
from torchsummary import summary
import numpy as np
from torch.utils.data import Dataset,DataLoader

def patchify(images, n_patches):
    n,c,h,w = images.shape
    patches = torch.zeros(n,n_patches**2, h*w//n_patches**2) #(34000, 100,25)
    patch_size = h//n_patches

    for idx,image in enumerate(images):
        for i in range(n_patches):
            for j in range(n_patches):
                patch = image[:,i*patch_size:(i+1)*patch_size,j*patch_size:(j+1)*patch_size]
                patches[idx,i*n_patches+j] = patch.flatten()
    return patches

def get_positional_embeddings(sequence_length,d):
    result = torch.ones(sequence_length,d)
    for i in range(sequence_length):
        for j in range(d):
            result[i][j] = np.sin(i/(10000**(j/d))) if j%2==0 else np.cos(i/(10000**((j-1)/d)))
    return result

class MSA(nn.Module):
    def __init__(self, d, n_heads=2):
        super(MSA, self).__init__()
        self.d = d
        self.n_heads = n_heads

        d_head = int(d / n_heads)
        self.q_mappings = nn.ModuleList([nn.Linear(d_head, d_head) for _ in range(self.n_heads)])
        self.k_mappings = nn.ModuleList([nn.Linear(d_head, d_head) for _ in range(self.n_heads)])
        self.v_mappings = nn.ModuleList([nn.Linear(d_head, d_head) for _ in range(self.n_heads)])
        self.d_head = d_head
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, sequences):
        # Sequences has shape (N, seq_length, token_dim)
        # We go into shape    (N, seq_length, n_heads, token_dim / n_heads)
        # And come back to    (N, seq_length, item_dim)  (through concatenation)
        result = []
        for sequence in sequences:
            seq_result = []
            for head in range(self.n_heads):
                q_mapping = self.q_mappings[head]
                k_mapping = self.k_mappings[head]
                v_mapping = self.v_mappings[head]

                seq = sequence[:, head * self.d_head: (head + 1) * self.d_head]
                q, k, v = q_mapping(seq), k_mapping(seq), v_mapping(seq)

                attention = self.softmax(q @ k.T / (self.d_head ** 0.5))
                seq_result.append(attention @ v)
            result.append(torch.hstack(seq_result))
        return torch.cat([torch.unsqueeze(r, dim=0) for r in result])

class ViTBlock(nn.Module):
    def __init__(self, hidden_d, n_heads, mlp_ratio=4):
        super(ViTBlock, self).__init__()
        self.hidden_d = hidden_d
        self.n_heads = n_heads

        self.norm1 = nn.LayerNorm(hidden_d)
        self.mhsa = MSA(hidden_d, n_heads)
        self.norm2 = nn.LayerNorm(hidden_d)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_d, mlp_ratio * hidden_d),
            nn.GELU(),
            nn.Linear(mlp_ratio * hidden_d, hidden_d)
        )

    def forward(self, x):
        out = x + self.mhsa(self.norm1(x))
        out = out + self.mlp(self.norm2(out))
        return out

class tang_vit(nn.Module):
    def __init__(self,chw = (1,50,50),n_patches =10,n_blocks=4,hidden_d=12, n_heads=2):
        super(tang_vit,self).__init__()

        self.chw = chw
        self.n_patches = n_patches
        self.n_blocks = n_blocks
        self.n_heads = n_heads
        self.hidden_d = hidden_d

        self.patch_size = (chw[1]/n_patches,chw[2]/n_patches)

        
        #linear mapping of patches (could also convolve)
        self.input_d = int(chw[0]*self.patch_size[0]*self.patch_size[1])
        self.linear_mapper = nn.Linear(self.input_d,self.hidden_d) #runnig 1:1 map from (3400,100,25) through a (25,12) mapper. so only happens on last dim

        #class token (learnable)
        #self.class_token = nn.Parameter(torch.rand(1,self.hidden_d))

        #pos embed
        self.pos_embed = nn.Parameter(torch.tensor(get_positional_embeddings(self.n_patches**2,self.hidden_d)))
        self.pos_embed.requires_grad=False

        #transformer blocks
        self.blocks = nn.ModuleList([ViTBlock(hidden_d,n_heads) for _ in range(n_blocks)])

        #flatten
        self.flatten = nn.Flatten()

        #linear
        self.linear = nn.Linear((n_patches**2) *hidden_d,1) #nn.Linear(self.hidden_d,1)

    def forward(self, images):
        n,c,h,w = images.shape
        patches = patchify(images, self.n_patches)
        tokens = self.linear_mapper(patches)

        #We can now add a parameter to our model and convert our (N, 100, 12) tokens tensor to an (N, 101, 12) tensor (we add the special token to each sequence).
        #added at front
        #tokens = torch.stack([torch.vstack((self.class_token,tokens[i])) for i in range(len(tokens))])

        #add pos embed
        pos_embed = self.pos_embed.repeat(n,1,1) #tokens have size (3400, 101, 12)
        out = tokens+pos_embed

        #transformer
        for block in self.blocks:
            out = block(out)
        
        out = self.flatten(out)
        return self.linear(out)

class ImageDataset(Dataset):
    def __init__(self, data, labels, num_neurons):
        self.data = data
        self.labels = labels
        self.num_neurons = num_neurons

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        img = torch.tensor(self.data[index], dtype=torch.float)
        label = torch.tensor(self.labels[index, 0:self.num_neurons], dtype=torch.float)
        return img, label
    

##################################################################################################################

from GS_functions import GF

train_img = np.load('../all_sites_data_prepared/pics_data/train_img_m1s1.npy')
train_resp = np.load('../all_sites_data_prepared/New_response_data/trainRsp_m1s1.npy')
train_img=np.reshape(train_img,(34000,1,50,50))
train_dataset = ImageDataset(train_img,train_resp,1)
train_loader = DataLoader(train_dataset,16,shuffle=True)

val_img = np.load('../all_sites_data_prepared/pics_data/val_img_m1s1.npy')
val_resp = np.load('../all_sites_data_prepared/New_response_data/valRsp_m1s1.npy')
val_img = np.reshape(val_img,(1000,1,50,50))
val_dataset = ImageDataset(val_img,val_resp,1)
val_loader = DataLoader(val_dataset,16,shuffle=True)#

MAXCORRSOFAR = 0

#(n_patches, n_blocks, hidden_d, n_heads)
for a,b,c,d in [(5,4,100,10),(5,4,200,10),(5,4,400,10),(5,4,800,10)]:

    savepathname=f'pbdh_{a}_{b}_{c}_{d}'
    GF.mkdir('./results-hiddend/',savepathname)
    log = open(f'./results-hiddend/pbdh_{a}_{b}_{c}_{d}/log.txt', "w")

    net = tang_vit(n_patches=a,n_blocks=b,hidden_d=c,n_heads=d)
    opt = torch.optim.Adam(net.parameters(),lr=0.0001)
    lfunc = torch.nn.MSELoss()

    losses = []
    accs = []
    for epoch in range(50):
        net.train()
        train_losses=[]
        for x,y in train_loader:
            l = lfunc(net(x),y)
            opt.zero_grad()
            l.backward()
            opt.step()
            train_losses.append(l.item())
        losses.append(np.mean(train_losses))
        
        val_losses=[]
        with torch.no_grad():
            net.eval()
            for x,y in val_loader:
                l = lfunc(net(x),y)
                val_losses.append(l.item())
        accs.append(np.mean(val_losses))

        log.write("Epoch " + str(epoch) + " train loss is:" + str(losses[-1])+"\n")
        log.write("Epoch " + str(epoch) + " val loss is:" + str(accs[-1])+"\n")


        images = np.asarray([item[0:1] for item in val_resp])
        responses = torch.from_numpy(np.float32(val_img))
        with torch.no_grad():
            net.eval()
            responses2 = net(responses).detach().numpy()
        images = np.transpose(images)
        responses2 = np.transpose(responses2)
        corrs = []
        for i in range(1):
            corrs.append(np.corrcoef(images[i], responses2[i])[0][1])
        # log.write("Correlation is " + str(np.mean(corrs))+"\n")
        log.write("Epoch" + str(epoch) + " correlation is " + str(np.mean(corrs))+"\n")


        if np.mean(corrs) > MAXCORRSOFAR:
            MAXCORRSOFAR=np.mean(corrs)
            log.write("Epoch " + str(epoch) + ": saved sucessfully.\n")
            torch.save(net.state_dict(),  f'./results-hiddend/pbdh_{a}_{b}_{c}_{d}/model_test.pth')
        log.write("\n")

    log.close()