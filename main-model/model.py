import torch
import torch.nn as nn
import torch.nn.functional as F

class MCEWithDirectionPenalty(nn.Module):
    def __init__(self, penalty_factor=1.0):
        super(MCEWithDirectionPenalty, self).__init__()
        self.penalty_factor = penalty_factor  # Controls the penalty strength
    
    def forward(self, predictions, targets, inputs):
        # Compute the Absolute Mean Cubed Error (aMCE)
        error = predictions - targets
        mce_loss = torch.mean(torch.pow(error, 4))
        
        # Handle direction mismatch penalties
        batch_size = predictions.shape[0]
        seq_len = predictions.shape[1]
        
        # Get the price column (assuming index 3 is the price)
        price_idx = 3
        
        # For first day predictions
        todays_price = inputs[:, -1, price_idx]  # Last input price
        
        # Initialize tensor to accumulate direction mismatches for all samples in batch
        direction_mismatch = torch.zeros(batch_size, device=predictions.device)
        
        # Check direction for first day
        tomorrow_prediction = predictions[:, 0, price_idx]
        tomorrow_actual = targets[:, 0, price_idx]
        
        # Calculate direction mismatch for first day
        pred_direction_up = tomorrow_prediction > todays_price
        actual_direction_up = tomorrow_actual > todays_price
        direction_mismatch += (pred_direction_up != actual_direction_up).float()
        
        # Check direction for remaining days
        for day in range(1, seq_len):
            prev_price = targets[:, day-1, price_idx]  # Use previous day's actual price
            current_pred = predictions[:, day, price_idx]
            current_actual = targets[:, day, price_idx]
            
            pred_direction_up = current_pred > prev_price
            actual_direction_up = current_actual > prev_price
            direction_mismatch += (pred_direction_up != actual_direction_up).float()
        
        # Normalize by sequence length to get average direction mismatch per sample
        direction_mismatch = direction_mismatch / seq_len
        # Take mean across batch
        avg_direction_mismatch = torch.mean(direction_mismatch)
        
        # Add penalty to the loss (scaled by penalty_factor)
        total_loss = mce_loss + self.penalty_factor * avg_direction_mismatch
        
        return total_loss

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
            x = F.dropout(x, p=0.5) # if dropout is needed to prevent overfitting we can add it here
        
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
    
    