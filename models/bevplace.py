import torch.nn as nn

from models.netvlad import NetVLAD
from models.groupnet import GroupNet



class BEVPlace(nn.Module):
    def __init__(self):
        super(BEVPlace, self).__init__()
        self.encoder = GroupNet()
        self.netvlad = NetVLAD()

    def forward(self, input):
        local_feature = self.encoder(input) 
        local_feature = local_feature.permute(0,2,1).unsqueeze(-1)
        global_feature = self.netvlad(local_feature) 

        return global_feature
