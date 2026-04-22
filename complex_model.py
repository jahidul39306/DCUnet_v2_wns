import torch
import torch.nn as nn
import torch.nn.functional as F

class ComplexConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ComplexConv2d, self).__init__()
        self.conv_real = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.conv_imag = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, real, imag):
        out_real = self.conv_real(real) - self.conv_imag(imag)
        out_imag = self.conv_real(imag) + self.conv_imag(real)
        return out_real, out_imag

class ComplexConvTranspose2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=2, stride=2, padding=0):
        super(ComplexConvTranspose2d, self).__init__()
        self.conv_t_real = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding)
        self.conv_t_imag = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, real, imag):
        # Matching exact sign convention of forward ComplexConv2d
        # to guarantee Skip-Connection phase geometry doesn't destructively interfere
        out_real = self.conv_t_real(real) - self.conv_t_imag(imag)
        out_imag = self.conv_t_real(imag) + self.conv_t_imag(real)
        return out_real, out_imag

class ComplexBatchNorm2d(nn.Module):
    def __init__(self, num_features):
        super(ComplexBatchNorm2d, self).__init__()
        self.bn_real = nn.BatchNorm2d(num_features)
        self.bn_imag = nn.BatchNorm2d(num_features)

    def forward(self, real, imag):
        return self.bn_real(real), self.bn_imag(imag)

class ComplexLeakyReLU(nn.Module):
    def __init__(self, negative_slope=0.2):
        super(ComplexLeakyReLU, self).__init__()
        self.relu = nn.LeakyReLU(negative_slope)
        
    def forward(self, real, imag):
        return self.relu(real), self.relu(imag)

class ComplexDoubleConv(nn.Module):
    """(ComplexConv2d -> ComplexBatchNorm -> ComplexLeakyReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = ComplexConv2d(in_channels, out_channels)
        self.bn1 = ComplexBatchNorm2d(out_channels)
        self.relu1 = ComplexLeakyReLU()
        
        self.conv2 = ComplexConv2d(out_channels, out_channels)
        self.bn2 = ComplexBatchNorm2d(out_channels)
        self.relu2 = ComplexLeakyReLU()

    def forward(self, real, imag):
        r1, i1 = self.conv1(real, imag)
        r1, i1 = self.bn1(r1, i1)
        r1, i1 = self.relu1(r1, i1)
        
        r2, i2 = self.conv2(r1, i1)
        r2, i2 = self.bn2(r2, i2)
        return self.relu2(r2, i2)

class ComplexDown(nn.Module):


    # instead of maxpooling we here used strided convolution to downsample the feature maps to avoid loss of phase information
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.downsample = ComplexConv2d(in_channels, in_channels, kernel_size=2, stride=2, padding=0)
        self.double_conv = ComplexDoubleConv(in_channels, out_channels)

    def forward(self, real, imag):
        d_real, d_imag = self.downsample(real, imag)
        return self.double_conv(d_real, d_imag)

class ComplexUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = ComplexConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = ComplexDoubleConv(in_channels, out_channels)

    def forward(self, real_en, imag_en, real_up, imag_up):
        r_upsampled, i_upsampled = self.up(real_up, imag_up)
        
        # Padding to fix mismatch like in regular U-Net
        diffY = real_en.size()[2] - r_upsampled.size()[2]
        diffX = real_en.size()[3] - r_upsampled.size()[3]
        
        r_upsampled = F.pad(r_upsampled, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        i_upsampled = F.pad(i_upsampled, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        
        # Concatenate skip connections
        r_cat = torch.cat([real_en, r_upsampled], dim=1)
        i_cat = torch.cat([imag_en, i_upsampled], dim=1)
        
        return self.conv(r_cat, i_cat)

class DeepComplexUNet(nn.Module):
 
    def __init__(self, n_channels=1):
        super(DeepComplexUNet, self).__init__()
        
        # Encoder (4 Downsampling levels = 5 Blocks Total)
        self.inc = ComplexDoubleConv(n_channels, 32)
        self.down1 = ComplexDown(32, 64)
        self.down2 = ComplexDown(64, 128)
        self.down3 = ComplexDown(128, 256)
        self.down4 = ComplexDown(256, 512)
        
        # Decoder (Massive Depth, Symmetric Scaling)
        self.up1 = ComplexUp(512, 256)
        self.up2 = ComplexUp(256, 128)
        self.up3 = ComplexUp(128, 64)
        self.up4 = ComplexUp(64, 32)
        
        # Final Output Layer 
        self.out_conv = ComplexConv2d(32, 1, kernel_size=1, padding=0)

    def forward(self, real, imag):
        if real.dim() == 3:
            real = real.unsqueeze(1)
            imag = imag.unsqueeze(1)
            
        # Encoder
        r1, i1 = self.inc(real, imag)     # Output: 32
        r2, i2 = self.down1(r1, i1)       # Output: 64
        r3, i3 = self.down2(r2, i2)       # Output: 128
        r4, i4 = self.down3(r3, i3)       # Output: 256
        r_latent, i_latent = self.down4(r4, i4) # Output: 512
        
        # Decoder Reconstructions
        r_up1, i_up1 = self.up1(r4, i4, r_latent, i_latent) # 512 -> 256
        r_up2, i_up2 = self.up2(r3, i3, r_up1, i_up1)       # 256 -> 128
        r_up3, i_up3 = self.up3(r2, i2, r_up2, i_up2)       # 128 -> 64
        r_up4, i_up4 = self.up4(r1, i1, r_up3, i_up3)       # 64 -> 32
        
        # Final Complex Mask prediction
        mask_real, mask_imag = self.out_conv(r_up4, i_up4)
        
        mask_real = torch.tanh(mask_real)
        mask_imag = torch.tanh(mask_imag)
        
        return mask_real, mask_imag
