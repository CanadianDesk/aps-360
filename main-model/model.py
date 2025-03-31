import torch
import torch.nn as nn
import torch.nn.functional as F

class EquityModel(nn.Module):
    def __init__(self, width_k_bar=18, kernel_size=3, input_height=256, output_height=16, conv_out_channel_list=[4, 16, 64, 256], pool_type='avg'):
        super(EquityModel, self).__init__()

        self.conv_layers = []

        # Create convolutional layers
        for i in range(len(conv_out_channel_list)):
            if i == 0:
                self.conv_layers.append(nn.Conv2d(1, conv_out_channel_list[i], kernel_size, padding=(2, 2), stride=1))
            else:
                self.conv_layers.append(nn.Conv2d(conv_out_channel_list[i-1], conv_out_channel_list[i], kernel_size, padding=(2,2), stride=1))
        self.final_conv_layer = nn.Conv2d(conv_out_channel_list[-1], 1, kernel_size-2, padding=1, stride=1) # Final conv layer to reduce channels to 1

        self.conv_layers = nn.ModuleList(self.conv_layers)
        self.width_k_bar = width_k_bar
        self.kernel_size = kernel_size
        self.input_width = input_height
        self.output_width = output_height
        self.conv_out_channel_list = conv_out_channel_list
        self.pool_type = pool_type
        # Calculate output height after all convolutions and pooling
        self.output_height = input_height
        for i in range(len(conv_out_channel_list)):
            self.output_height = (self.output_height - kernel_size + 2) // 2
        # Final output height after all convolutions and pooling
        self.output_height = self.output_height - kernel_size + 1
        # Check if the final output height matches the expected output height
        # if self.output_height != output_height:
        #     raise ValueError("The calculated output height {} does not match the expected output height {}.".format(self.output_height, output_height))
        # Check if the input width is divisible by the kernel size
        
    def forward(self, x):
        
        x = x.unsqueeze(1) # add channel dimension, x.shape = (batch_size, 1, input_height, width_k_bar)
        for conv in self.conv_layers:

            x = conv(x)

            # x = F.relu(x) # Perhaps try other activations like LeakyReLU, or just linear
            x = F.dropout(x, p=0.65) # if dropout is needed to prevent overfitting we can add it here
        
            # Pooling layer
            if self.pool_type == 'avg':
                x = F.avg_pool2d(x, kernel_size=(2, 1), stride=(2, 1), padding=(0, 0))
            elif self.pool_type == 'max':
                x = F.max_pool2d(x, kernel_size=(2, 1), stride=(2, 1), padding=(0, 0))
            else: 
                raise ValueError("pool_type must be 'avg' or 'max', got {}".format(self.pool_type))
            
        x = self.final_conv_layer(x)

        x = x.squeeze(1) # remove channel dimension, x.shape = (batch_size, output_height, width_k_bar)
        return x # x.shape = (batch_size, output_height, width_k_bar)
    
    