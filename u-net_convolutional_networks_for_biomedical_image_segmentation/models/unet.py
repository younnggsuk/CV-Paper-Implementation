import torch
import torch.nn as nn


__all__ = ['unet', 'UNet', 'UNet_BN']


class CBR(nn.Module):

    def __init__(self, c_in, c_out, k_size, stride, padding):
        super().__init__()
        
        self.conv = nn.Conv2d(in_channels=c_in, out_channels=c_out, kernel_size=k_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(num_features=c_out)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)

        return x


class CR(nn.Module):

    def __init__(self, c_in, c_out, k_size, stride, padding):
        super().__init__()
        
        self.conv = nn.Conv2d(in_channels=c_in, out_channels=c_out, kernel_size=k_size, stride=stride, padding=padding)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.act(x)

        return x


class UNet(nn.Module):

    def __init__(self, in_channels, num_classes):
        super().__init__()

        # Contracting path
        self.encoder_1_1 = CR(in_channels, 64, k_size=3, stride=1, padding=1)
        self.encoder_1_2 = CR(64, 64, k_size=3, stride=1, padding=1)
        self.encoder_1_3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.encoder_2_1 = CR(64, 128, k_size=3, stride=1, padding=1)
        self.encoder_2_2 = CR(128, 128, k_size=3, stride=1, padding=1)
        self.encoder_2_3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.encoder_3_1 = CR(128, 256, k_size=3, stride=1, padding=1)
        self.encoder_3_2 = CR(256, 256, k_size=3, stride=1, padding=1)
        self.encoder_3_3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.encoder_4_1 = CR(256, 512, k_size=3, stride=1, padding=1)
        self.encoder_4_2 = CR(512, 512, k_size=3, stride=1, padding=1)
        self.encoder_4_3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.encoder_5_1 = CR(512, 1024, k_size=3, stride=1, padding=1)
        self.encoder_5_2 = CR(1024, 1024, k_size=3, stride=1, padding=1)

        # Expansive path
        self.decoder_1_1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2, padding=0)
        self.decoder_1_2 = CR(1024, 512, k_size=3, stride=1, padding=1)
        self.decoder_1_3 = CR(512, 512, k_size=3, stride=1, padding=1)

        self.decoder_2_1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2, padding=0)
        self.decoder_2_2 = CR(512, 256, k_size=3, stride=1, padding=1)
        self.decoder_2_3 = CR(256, 256, k_size=3, stride=1, padding=1)

        self.decoder_3_1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2, padding=0)
        self.decoder_3_2 = CR(256, 128, k_size=3, stride=1, padding=1)
        self.decoder_3_3 = CR(128, 128, k_size=3, stride=1, padding=1)

        self.decoder_4_1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2, padding=0)
        self.decoder_4_2 = CR(128, 64, k_size=3, stride=1, padding=1)
        self.decoder_4_3 = CR(64, 64, k_size=3, stride=1, padding=1)

        self.pointwise_conv = nn.Conv2d(64, num_classes, kernel_size=1, stride=1, padding=0)
        
        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

    def forward(self, x):

        # Contracting path
        enc_1_1 = self.encoder_1_1(x)
        enc_1_2 = self.encoder_1_2(enc_1_1)
        pool_1 = self.encoder_1_3(enc_1_2)

        enc_2_1 = self.encoder_2_1(pool_1)
        enc_2_2 = self.encoder_2_2(enc_2_1)
        pool_2 = self.encoder_2_3(enc_2_2)

        enc_3_1 = self.encoder_3_1(pool_2)
        enc_3_2 = self.encoder_3_2(enc_3_1)
        pool_3 = self.encoder_3_3(enc_3_2)

        enc_4_1 = self.encoder_4_1(pool_3)
        enc_4_2 = self.encoder_4_2(enc_4_1)
        pool_4 = self.encoder_4_3(enc_4_2)

        enc_5_1 = self.encoder_5_1(pool_4)
        enc_5_2 = self.encoder_5_2(enc_5_1)

        # Expansive path
        up_conv_1 = self.decoder_1_1(enc_5_2, output_size=enc_4_1.shape[2:])
        dec_1_2 = self.decoder_1_2(torch.cat([enc_4_2, up_conv_1], dim=1))
        dec_1_3 = self.decoder_1_3(dec_1_2)

        up_conv_2 = self.decoder_2_1(dec_1_3, output_size=enc_3_1.shape[2:])
        dec_2_2 = self.decoder_2_2(torch.cat([enc_3_2, up_conv_2], dim=1))
        dec_2_3 = self.decoder_2_3(dec_2_2)

        up_conv_3 = self.decoder_3_1(dec_2_3, output_size=enc_2_1.shape[2:])
        dec_3_2 = self.decoder_3_2(torch.cat([enc_2_2, up_conv_3], dim=1))
        dec_3_3 = self.decoder_3_3(dec_3_2)

        up_conv_4 = self.decoder_4_1(dec_3_3, output_size=enc_1_1.shape[2:])
        dec_4_2 = self.decoder_4_2(torch.cat([enc_1_2, up_conv_4], dim=1))
        dec_4_3 = self.decoder_4_3(dec_4_2)
        
        out = self.pointwise_conv(dec_4_3)

        return out


class UNet_BN(nn.Module):

    def __init__(self, in_channels, num_classes):
        super().__init__()
        
        # Contracting path
        self.encoder_1_1 = CBR(in_channels, 64, k_size=3, stride=1, padding=1)
        self.encoder_1_2 = CBR(64, 64, k_size=3, stride=1, padding=1)
        self.encoder_1_3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.encoder_2_1 = CBR(64, 128, k_size=3, stride=1, padding=1)
        self.encoder_2_2 = CBR(128, 128, k_size=3, stride=1, padding=1)
        self.encoder_2_3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.encoder_3_1 = CBR(128, 256, k_size=3, stride=1, padding=1)
        self.encoder_3_2 = CBR(256, 256, k_size=3, stride=1, padding=1)
        self.encoder_3_3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.encoder_4_1 = CBR(256, 512, k_size=3, stride=1, padding=1)
        self.encoder_4_2 = CBR(512, 512, k_size=3, stride=1, padding=1)
        self.encoder_4_3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.encoder_5_1 = CBR(512, 1024, k_size=3, stride=1, padding=1)
        self.encoder_5_2 = CBR(1024, 1024, k_size=3, stride=1, padding=1)

        # Expansive path
        self.decoder_1_1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2, padding=0)
        self.decoder_1_2 = CBR(1024, 512, k_size=3, stride=1, padding=1)
        self.decoder_1_3 = CBR(512, 512, k_size=3, stride=1, padding=1)

        self.decoder_2_1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2, padding=0)
        self.decoder_2_2 = CBR(512, 256, k_size=3, stride=1, padding=1)
        self.decoder_2_3 = CBR(256, 256, k_size=3, stride=1, padding=1)

        self.decoder_3_1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2, padding=0)
        self.decoder_3_2 = CBR(256, 128, k_size=3, stride=1, padding=1)
        self.decoder_3_3 = CBR(128, 128, k_size=3, stride=1, padding=1)

        self.decoder_4_1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2, padding=0)
        self.decoder_4_2 = CBR(128, 64, k_size=3, stride=1, padding=1)
        self.decoder_4_3 = CBR(64, 64, k_size=3, stride=1, padding=1)

        self.pointwise_conv = nn.Conv2d(64, num_classes, kernel_size=1, stride=1, padding=0)
        
        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

    def forward(self, x):

        # Contracting path
        enc_1_1 = self.encoder_1_1(x)
        enc_1_2 = self.encoder_1_2(enc_1_1)
        pool_1 = self.encoder_1_3(enc_1_2)

        enc_2_1 = self.encoder_2_1(pool_1)
        enc_2_2 = self.encoder_2_2(enc_2_1)
        pool_2 = self.encoder_2_3(enc_2_2)

        enc_3_1 = self.encoder_3_1(pool_2)
        enc_3_2 = self.encoder_3_2(enc_3_1)
        pool_3 = self.encoder_3_3(enc_3_2)

        enc_4_1 = self.encoder_4_1(pool_3)
        enc_4_2 = self.encoder_4_2(enc_4_1)
        pool_4 = self.encoder_4_3(enc_4_2)

        enc_5_1 = self.encoder_5_1(pool_4)
        enc_5_2 = self.encoder_5_2(enc_5_1)

        # Expansive path
        up_conv_1 = self.decoder_1_1(enc_5_2, output_size=enc_4_1.shape[2:])
        dec_1_2 = self.decoder_1_2(torch.cat([enc_4_2, up_conv_1], dim=1))
        dec_1_3 = self.decoder_1_3(dec_1_2)

        up_conv_2 = self.decoder_2_1(dec_1_3, output_size=enc_3_1.shape[2:])
        dec_2_2 = self.decoder_2_2(torch.cat([enc_3_2, up_conv_2], dim=1))
        dec_2_3 = self.decoder_2_3(dec_2_2)

        up_conv_3 = self.decoder_3_1(dec_2_3, output_size=enc_2_1.shape[2:])
        dec_3_2 = self.decoder_3_2(torch.cat([enc_2_2, up_conv_3], dim=1))
        dec_3_3 = self.decoder_3_3(dec_3_2)

        up_conv_4 = self.decoder_4_1(dec_3_3, output_size=enc_1_1.shape[2:])
        dec_4_2 = self.decoder_4_2(torch.cat([enc_1_2, up_conv_4], dim=1))
        dec_4_3 = self.decoder_4_3(dec_4_2)
        
        out = self.pointwise_conv(dec_4_3)

        return out


def unet(in_channels, num_classes, use_batchnorm=False):
    if use_batchnorm:
        return UNet_BN(in_channels, num_classes)
    else:
        return UNet(in_channels, num_classes)