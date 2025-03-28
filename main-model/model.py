import torch
import torch.nn as nn
import torch.nn.functional as F

class EquityModel(nn.Module):
    def __init__(self, width_k_bar=8, kernel_size=2, input_height=256, output_height=16, conv_out_channel_list=[4, 16, 64, 256, 32, 1], avg_pool_freq=2):
        super(EquityModel, self).__init__()

        self.conv_layers = []
        for i in range(len(conv_out_channel_list)):
            if i == 0:
                self.conv_layers.append(nn.Conv2d(1, conv_out_channel_list[i], kernel_size, padding=0, stride=1))
            else:
                self.conv_layers.append(nn.Conv2d(conv_out_channel_list[i-1], conv_out_channel_list[i], kernel_size, padding=0, stride=1))
        self.conv_layers = nn.ModuleList(self.conv_layers)
        self.width_k_bar = width_k_bar
        self.kernel_size = kernel_size
        self.input_width = input_height
        self.output_width = output_height
        self.conv_out_channel_list = conv_out_channel_list
        self.avg_pool_freq = avg_pool_freq
        
    def forward(self, x):

        x = x.unsqueeze(1) # add channel dimension, x.shape = (batch_size, 1, input_height, width_k_bar)
        count = 0
        for conv in self.conv_layers:
            x = conv(x)
            count += 1
            if count % self.avg_pool_freq == 0:
                x = F.avg_pool2d(x, kernel_size=(1, 2), stride=(1, 2), padding=(0, 0))

        return x # (batch_size, conv_out_channel_list[-1], output_height, width_k_bar)
    
    