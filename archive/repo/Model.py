import torch
import torch.nn as nn

def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True)
    )   


class UNet_mt(nn.Module):

    def __init__(self, ch, n_class, use_cuda = True, first_number_of_filter = 64):
        super().__init__()
        self.use_cuda = use_cuda

        self.conv_layer_encoder1 = double_conv(ch, first_number_of_filter)
        self.conv_layer_encoder2 = double_conv(first_number_of_filter, first_number_of_filter*2)             
        self.conv_layer_encoder3 = double_conv(first_number_of_filter*2, first_number_of_filter*4)             
        self.conv_layer_encoder4 = double_conv(first_number_of_filter*4, first_number_of_filter*8)    

        self.down_sample = nn.MaxPool2d(2)    
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)        

        self.conv_layer_reg_decoder3 = double_conv(256 + 512, 256)
        self.conv_layer_reg_decoder2 = double_conv(128 + 256, 128)            
        self.conv_layer_reg_decoder1 = double_conv(128 + 64, 64)            
        self.conv_reg_last = nn.Conv2d(64, n_class, 1)

        self.conv_layer_bc_decoder3 = double_conv(256 + 512, 256)
        self.conv_layer_bc_decoder2 = double_conv(128 + 256, 128)            
        self.conv_layer_bc_decoder1 = double_conv(128 + 64, 64)            
        self.conv_bc_last = nn.Conv2d(64, n_class, 1)
        

    def forward(self, x):
        conv1 = self.conv_layer_encoder1(x)
        #W, H, 64
        x = self.down_sample(conv1)
        #W/2, H/2, 64

        conv2 = self.conv_layer_encoder2(x)
        #W/2, H/2, 128
        x = self.down_sample(conv2)
        #W/4, H/4, 128

        conv3 = self.conv_layer_encoder3(x)
        #W/4, H/4, 256
        x = self.down_sample(conv3)   
        #W/8, H/8, 256

        x = self.conv_layer_encoder4(x)
        #W/8, H/8, 512

        ####Decoder1
        x1 = self.upsample(x)  
        #W/4, H/4, 512 
        x1 = torch.cat([x1, conv3], dim=1)
        #W/4, H/4, 512 +256
        
        x1 = self.conv_layer_reg_decoder3(x1)
        #W/4, H/4, 256
        x1 = self.upsample(x1)  

        #W/2, H/2, 256
        x1 = torch.cat([x1, conv2], dim=1)       
        #W/2, H/2, 256 + 128
        x1 = self.conv_layer_reg_decoder2(x1)
        #W/2, H/2, 128

        x1 = self.upsample(x1)   
        #W, H, 128
        x1 = torch.cat([x1, conv1], dim=1)   
        #W, H, 128 + 64
        x1 = self.conv_layer_reg_decoder1(x1)
        #W, H, 64

        #no sigmoid, just return score
        out_reg = self.conv_reg_last(x1)
        #W, H, 1

        ####Decoder2
        x2 = self.upsample(x)  
        #W/4, H/4, 512 
        x2 = torch.cat([x2, conv3], dim=1)
        #W/4, H/4, 512 +256
        
        x2 = self.conv_layer_bc_decoder3(x2)
        #W/4, H/4, 256
        x2 = self.upsample(x2)  

        #W/2, H/2, 256
        x2 = torch.cat([x2, conv2], dim=1)       
        #W/2, H/2, 256 + 128
        x2 = self.conv_layer_bc_decoder2(x2)
        #W/2, H/2, 128

        x2 = self.upsample(x2)   
        #W, H, 128
        x2 = torch.cat([x2, conv1], dim=1)   
        #W, H, 128 + 64
        x2 = self.conv_layer_bc_decoder1(x2)
        #W, H, 64

        out_bc = self.conv_bc_last(x2)
        #W, H, 1

        return out_bc, out_reg



class UNet(nn.Module):

    def __init__(self, ch, n_class, use_cuda = True, first_number_of_filter = 64):
        super().__init__()
        self.use_cuda = use_cuda

        self.conv_layer_encoder1 = double_conv(ch, first_number_of_filter)
        self.conv_layer_encoder2 = double_conv(first_number_of_filter, first_number_of_filter*2)             
        self.conv_layer_encoder3 = double_conv(first_number_of_filter*2, first_number_of_filter*4)             
        self.conv_layer_encoder4 = double_conv(first_number_of_filter*4, first_number_of_filter*8)    

        self.down_sample = nn.MaxPool2d(2)    
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)        

        self.conv_layer_decoder3 = double_conv(256 + 512, 256)
        self.conv_layer_decoder2 = double_conv(128 + 256, 128)            
        self.conv_layer_decoder1 = double_conv(128 + 64, 64)            
        self.conv_last = nn.Conv2d(64, n_class, 1)


        # self.conv_layer_reg_decoder3 = double_conv(256 + 512, 256)
        # self.conv_layer_reg_decoder2 = double_conv(128 + 256, 128)            
        # self.conv_layer_reg_decoder1 = double_conv(128 + 64, 64)            
        # self.conv_reg_last = nn.Conv2d(64, n_class, 1)

        # self.conv_layer_bc_decoder3 = double_conv(256 + 512, 256)
        # self.conv_layer_bc_decoder2 = double_conv(128 + 256, 128)            
        # self.conv_layer_bc_decoder1 = double_conv(128 + 64, 64)            
        # self.conv_bc_last = nn.Conv2d(64, n_class, 1)
        

    def forward(self, x):
        conv1 = self.conv_layer_encoder1(x)
        #W, H, 64
        x = self.down_sample(conv1)
        #W/2, H/2, 64

        conv2 = self.conv_layer_encoder2(x)
        #W/2, H/2, 128
        x = self.down_sample(conv2)
        #W/4, H/4, 128

        conv3 = self.conv_layer_encoder3(x)
        #W/4, H/4, 256
        x = self.down_sample(conv3)   
        #W/8, H/8, 256

        x = self.conv_layer_encoder4(x)
        #W/8, H/8, 512

        x = self.upsample(x)  
        #W/4, H/4, 512 
        x = torch.cat([x, conv3], dim=1)
        #W/4, H/4, 512 +256
        
        x = self.conv_layer_decoder3(x)
        #W/4, H/4, 256
        x = self.upsample(x)  

        #W/2, H/2, 256
        x = torch.cat([x, conv2], dim=1)       
        #W/2, H/2, 256 + 128
        x = self.conv_layer_decoder2(x)
        #W/2, H/2, 128

        x = self.upsample(x)   
        #W, H, 128
        x = torch.cat([x, conv1], dim=1)   
        #W, H, 128 + 64
        x = self.conv_layer_decoder1(x)
        #W, H, 64

        out = self.conv_last(x)
        #W, H, 1

        return out