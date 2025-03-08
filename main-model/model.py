import torch
import torch.nn as nn
import torch.nn.functional as F

class EquityModel(nn.Module):
    def __init__(self, height_k_bar=8, kernel_size=2, input_width=512, output_width=32, conv_out_channel_list=[32,64], fc_size_list=[128, 64]):
        super(EquityModel, self).__init__()

        self.conv_layers = []
        for i in range(len(conv_out_channel_list)):
            if i == 0:
                self.conv_layers.append(nn.Conv2d(1, conv_out_channel_list[i], kernel_size, padding=kernel_size//2, stride=1))
            else:
                self.conv_layers.append(nn.Conv2d(conv_out_channel_list[i-1], conv_out_channel_list[i], kernel_size, padding=kernel_size//2, stride=1))
        self.conv_layers = nn.ModuleList(self.conv_layers)

        self.fc_layers = []
        for i in range(len(fc_size_list)):
            if i == 0:
                self.fc_layers.append(nn.Linear(conv_out_channel_list[-1]*input_width*height_k_bar), fc_size_list[i])
            else:
                self.fc_layers.append(nn.Linear(fc_size_list[i-1], fc_size_list[i]))

        self.fc_layers.append(nn.Linear(fc_size_list[-1], output_width))
        self.fc_layers = nn.ModuleList(self.fc_layers)

    def forward(self, x):

        x = x.unsqueeze(1) # add channel dimension
        for conv in self.conv_layers:
            x = conv(x)
            x = F.relu(x)

        x = x.view(x.size(0), -1)
        for fc in self.fc_layers:
            x = fc(x)
            x = F.relu(x)

        return x
    