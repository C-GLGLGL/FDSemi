import torch.nn as nn
import torch.nn.functional as F


class AALDiscriminator(nn.Module):
    def __init__(self, num_classes, img_height = 512, img_weight = 512,):
        super(AALDiscriminator, self).__init__()
        self.conv = nn.Sequential(
            nn.Linear(img_height * img_weight * num_classes, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        #return self.conv(x.view(x.size(0), -1))
        return self.conv(x.view(x.size(0), -1)).view(-1)