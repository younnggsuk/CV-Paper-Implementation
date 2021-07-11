import torch
import torch.nn as nn


__all__ = ['se_resnet']


def conv1x1(in_channels, out_channels, is_activate=True):
    layers = [nn.Conv2d(in_channels, out_channels, 
                        kernel_size=1, stride=1,
                        padding=0, bias=False),
              nn.BatchNorm2d(out_channels)]
                           
    if is_activate:
        layers.append(nn.ReLU(inplace=True))

    return nn.Sequential(*layers)


def conv3x3(in_channels, out_channels, is_downsample=False):
    if is_downsample:
        stride = 2
    else:
        stride = 1
    
    return nn.Sequential(nn.Conv2d(in_channels, out_channels,
                                   kernel_size=3, stride=stride,
                                   padding=1, bias=False),
                         nn.BatchNorm2d(out_channels),
                         nn.ReLU(inplace=True))


def shortcut_proj(in_channels, out_channels, is_downsample=False):
    if in_channels == out_channels:
        return None

    if is_downsample:
        stride = 2
    else:
        stride = 1
    
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, 
                                   kernel_size=1, stride=stride, 
                                   padding=0, bias=False),
                         nn.BatchNorm2d(out_channels))


class SEResNetModule(nn.Module):
    
    def __init__(self, in_channels, out_channels, is_downsample=False, reduction_raio=16):
        super().__init__()
        
        # residual module
        self.residual = nn.Sequential(
            conv1x1(in_channels, out_channels // 4),
            conv3x3(out_channels // 4, out_channels // 4, is_downsample),
            conv1x1(out_channels // 4, out_channels, is_activate=False)
        )

        # shortcut projection
        self.shortcut_proj = shortcut_proj(in_channels, out_channels, is_downsample)

        # squeeze
        self.squeeze = nn.AdaptiveAvgPool2d(1)

        # excitation
        self.excitation = nn.Sequential(
            nn.Linear(out_channels, out_channels // reduction_raio, bias=False),
            nn.ReLU(),
            nn.Linear(out_channels // reduction_raio, out_channels, bias=False),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):            
        identity = x

        u = self.residual(x)

        z = self.squeeze(u)
        z = z.view(z.shape[:2])
        
        s = self.excitation(z)
        s = s.view(*s.shape, 1, 1)

        x_hat = torch.mul(u, s)

        if self.shortcut_proj:
            identity = self.shortcut_proj(x)
        
        x_hat += identity
        out = self.relu(x_hat)

        return out
        

class SEResNet50(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        # conv1
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # conv2_x
        self.conv2_x = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            SEResNetModule(in_channels=64, out_channels=256),
            *[SEResNetModule(in_channels=256, out_channels=256) for _ in range(1, 3)]
        )
        
        # conv3_x
        self.conv3_x = nn.Sequential(
            SEResNetModule(in_channels=256, out_channels=512, is_downsample=True),
            *[SEResNetModule(in_channels=512, out_channels=512) for _ in range(1, 4)]
        )
        
        # conv4_x
        self.conv4_x = nn.Sequential(
            SEResNetModule(in_channels=512, out_channels=1024, is_downsample=True),
            *[SEResNetModule(in_channels=1024, out_channels=1024) for _ in range(1, 6)]
        )
        
        # conv5_x
        self.conv5_x = nn.Sequential(
            SEResNetModule(in_channels=1024, out_channels=2048, is_downsample=True),
            *[SEResNetModule(in_channels=2048, out_channels=2048) for _ in range(1, 3)]
        )
        
        # average pool
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # 1000-d fc
        self.fc = nn.Linear(2048, 1000)

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        
        x = self.conv2_x(x)
        x = self.conv3_x(x)
        x = self.conv4_x(x)
        x = self.conv5_x(x)

        x = self.avg_pool(x)
        x = x.view(-1, 2048)
        x = self.fc(x)

        return x


class SEResNet101(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        # conv1
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # conv2_x
        self.conv2_x = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            SEResNetModule(in_channels=64, out_channels=256),
            *[SEResNetModule(in_channels=256, out_channels=256) for _ in range(1, 3)]
        )
        
        # conv3_x
        self.conv3_x = nn.Sequential(
            SEResNetModule(in_channels=256, out_channels=512, is_downsample=True),
            *[SEResNetModule(in_channels=512, out_channels=512) for _ in range(1, 4)]
        )
        
        # conv4_x
        self.conv4_x = nn.Sequential(
            SEResNetModule(in_channels=512, out_channels=1024, is_downsample=True),
            *[SEResNetModule(in_channels=1024, out_channels=1024) for _ in range(1, 23)]
        )
        
        # conv5_x
        self.conv5_x = nn.Sequential(
            SEResNetModule(in_channels=1024, out_channels=2048, is_downsample=True),
            *[SEResNetModule(in_channels=2048, out_channels=2048) for _ in range(1, 3)]
        )
        
        # average pool
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # 1000-d fc
        self.fc = nn.Linear(2048, 1000)

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        
        x = self.conv2_x(x)
        x = self.conv3_x(x)
        x = self.conv4_x(x)
        x = self.conv5_x(x)

        x = self.avg_pool(x)
        x = x.view(-1, 2048)
        x = self.fc(x)

        return x


class SEResNet152(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        # conv1
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # conv2_x
        self.conv2_x = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            SEResNetModule(in_channels=64, out_channels=256),
            *[SEResNetModule(in_channels=256, out_channels=256) for _ in range(1, 3)]
        )
        
        # conv3_x
        self.conv3_x = nn.Sequential(
            SEResNetModule(in_channels=256, out_channels=512, is_downsample=True),
            *[SEResNetModule(in_channels=512, out_channels=512) for _ in range(1, 8)]
        )
        
        # conv4_x
        self.conv4_x = nn.Sequential(
            SEResNetModule(in_channels=512, out_channels=1024, is_downsample=True),
            *[SEResNetModule(in_channels=1024, out_channels=1024) for _ in range(1, 36)]
        )
        
        # conv5_x
        self.conv5_x = nn.Sequential(
            SEResNetModule(in_channels=1024, out_channels=2048, is_downsample=True),
            *[SEResNetModule(in_channels=2048, out_channels=2048) for _ in range(1, 3)]
        )
        
        # average pool
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # 1000-d fc
        self.fc = nn.Linear(2048, 1000)

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        
        x = self.conv2_x(x)
        x = self.conv3_x(x)
        x = self.conv4_x(x)
        x = self.conv5_x(x)

        x = self.avg_pool(x)
        x = x.view(-1, 2048)
        x = self.fc(x)

        return x


def se_resnet(num_layers):
    if num_layers == 50: return SEResNet50()
    if num_layers == 101: return SEResNet101()
    if num_layers == 152: return SEResNet152()