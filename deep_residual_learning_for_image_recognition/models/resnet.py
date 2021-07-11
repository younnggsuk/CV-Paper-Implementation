import torch.nn as nn


__all__ = ['resnet']


class BasicBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels, is_downsample=False):
        super(BasicBlock, self).__init__()

        # stacked layers
        if is_downsample:
            self.conv1 = nn.Conv2d(in_channels,
                                   out_channels,
                                   kernel_size=3,
                                   stride=2,
                                   padding=1,
                                   bias=False)
        else:
            self.conv1 = nn.Conv2d(in_channels,
                                   out_channels,
                                   kernel_size=3,
                                   stride=1,
                                   padding=1,
                                   bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels,
                               out_channels,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # shortcut connection
        if in_channels == out_channels:
            self.shortcut_proj = None
        else:
            if is_downsample:
                self.shortcut_proj = nn.Sequential(
                    nn.Conv2d(in_channels,
                              out_channels,
                              kernel_size=1,
                              stride=2,
                              bias=False),
                    nn.BatchNorm2d(out_channels)
                )
            else:
                self.shortcut_proj = nn.Sequential(
                    nn.Conv2d(in_channels,
                              out_channels,
                              kernel_size=1,
                              stride=1,
                              bias=False),
                    nn.BatchNorm2d(out_channels)
                )
        
    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.shortcut_proj:
            identity = self.shortcut_proj(x)
        
        out += identity
        out = self.relu(out)
        
        return out


class BottleneckBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels, is_downsample=False):
        super(BottleneckBlock, self).__init__()
        
        # stacked layers
        self.conv1 = nn.Conv2d(in_channels,
                               out_channels // 4,
                               kernel_size=1,
                               stride=1,
                               padding=0,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels // 4)
        self.relu = nn.ReLU()
        if is_downsample:
            self.conv2 = nn.Conv2d(out_channels // 4,
                                   out_channels // 4,
                                   kernel_size=3,
                                   stride=2,
                                   padding=1,
                                   bias=False)
        else:
            self.conv2 = nn.Conv2d(out_channels // 4,
                                   out_channels // 4,
                                   kernel_size=3,
                                   stride=1,
                                   padding=1,
                                   bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels // 4)
        self.conv3 = nn.Conv2d(out_channels // 4,
                               out_channels,
                               kernel_size=1,
                               stride=1,
                               padding=0,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        
        # shortcut connection
        if in_channels == out_channels:
            self.shortcut_proj = None
        else:
            if is_downsample:
                self.shortcut_proj = nn.Sequential(
                    nn.Conv2d(in_channels,
                              out_channels,
                              kernel_size=1,
                              stride=2,
                              bias=False),
                    nn.BatchNorm2d(out_channels)
                )
            else:
                self.shortcut_proj = nn.Sequential(
                    nn.Conv2d(in_channels,
                              out_channels,
                              kernel_size=1,
                              stride=1,
                              bias=False),
                    nn.BatchNorm2d(out_channels)
                )
        
    def forward(self, x):
        identity = x
                    
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
                    
        if self.shortcut_proj:
            identity = self.shortcut_proj(x)
        
        out += identity
        out = self.relu(out)
        
        return out


class ResNet18(nn.Module):
    
    def __init__(self):
        super(ResNet18, self).__init__()

        # conv1
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # conv2_x
        self.conv2_x = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            BasicBlock(in_channels=64, out_channels=64),
            BasicBlock(in_channels=64, out_channels=64)
        )

        # conv3_x
        self.conv3_x = nn.Sequential(
            BasicBlock(in_channels=64, out_channels=128, is_downsample=True),
            BasicBlock(in_channels=128, out_channels=128)
        )

        # conv4_x
        self.conv4_x = nn.Sequential(
            BasicBlock(in_channels=128, out_channels=256, is_downsample=True),
            BasicBlock(in_channels=256, out_channels=256)
        )

        # conv5_x
        self.conv5_x = nn.Sequential(
            BasicBlock(in_channels=256, out_channels=512, is_downsample=True),
            BasicBlock(in_channels=512, out_channels=512)
        )

        # average pool
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # 1000-d fc
        self.fc = nn.Linear(512, 1000)

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu') 

    def forward(self, x):
        x = self.conv1(x)

        x = self.conv2_x(x)
        x = self.conv3_x(x)
        x = self.conv4_x(x)
        x = self.conv5_x(x)
        
        x = self.avg_pool(x)
        x = x.view(-1, 512)
        x = self.fc(x)

        return x


class ResNet34(nn.Module):
    
    def __init__(self):
        super(ResNet34, self).__init__()

        # conv1
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # conv2_x
        self.conv2_x = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            *[BasicBlock(in_channels=64, out_channels=64) for _ in range(3)]
        )

        # conv3_x
        self.conv3_x = nn.Sequential(
            BasicBlock(in_channels=64, out_channels=128, is_downsample=True),
            *[BasicBlock(in_channels=128, out_channels=128) for _ in range(1, 4)]
        )

        # conv4_x
        self.conv4_x = nn.Sequential(
            BasicBlock(in_channels=128, out_channels=256, is_downsample=True),
            *[BasicBlock(in_channels=256, out_channels=256) for _ in range(1, 6)]
        )

        # conv5_x
        self.conv5_x = nn.Sequential(
            BasicBlock(in_channels=256, out_channels=512, is_downsample=True),
            *[BasicBlock(in_channels=512, out_channels=512) for _ in range(1, 3)]
        )

        # average pool
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # 1000-d fc
        self.fc = nn.Linear(512, 1000)

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu') 

    def forward(self, x):
        x = self.conv1(x)

        x = self.conv2_x(x)
        x = self.conv3_x(x)
        x = self.conv4_x(x)
        x = self.conv5_x(x)

        x = self.avg_pool(x)
        x = x.view(-1, 512)
        x = self.fc(x)

        return x


class ResNet50(nn.Module):
    
    def __init__(self):
        super(ResNet50, self).__init__()
        
        # conv1
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # conv2_x
        self.conv2_x = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            BottleneckBlock(in_channels=64, out_channels=256),
            *[BottleneckBlock(in_channels=256, out_channels=256) for _ in range(1, 3)]
        )
        
        # conv3_x
        self.conv3_x = nn.Sequential(
            BottleneckBlock(in_channels=256, out_channels=512, is_downsample=True),
            *[BottleneckBlock(in_channels=512, out_channels=512) for _ in range(1, 4)]
        )
        
        # conv4_x
        self.conv4_x = nn.Sequential(
            BottleneckBlock(in_channels=512, out_channels=1024, is_downsample=True),
            *[BottleneckBlock(in_channels=1024, out_channels=1024) for _ in range(1, 6)]
        )
        
        # conv5_x
        self.conv5_x = nn.Sequential(
            BottleneckBlock(in_channels=1024, out_channels=2048, is_downsample=True),
            *[BottleneckBlock(in_channels=2048, out_channels=2048) for _ in range(1, 3)]
        )
        
        # average pool
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # 1000-d fc
        self.fc = nn.Linear(2048, 1000)

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu') 

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


class ResNet101(nn.Module):
    
    def __init__(self):
        super(ResNet101, self).__init__()
        
        # conv1
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # conv2_x
        self.conv2_x = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            BottleneckBlock(in_channels=64, out_channels=256),
            *[BottleneckBlock(in_channels=256, out_channels=256) for _ in range(1, 3)]
        )
        
        # conv3_x
        self.conv3_x = nn.Sequential(
            BottleneckBlock(in_channels=256, out_channels=512, is_downsample=True),
            *[BottleneckBlock(in_channels=512, out_channels=512) for _ in range(1, 4)]
        )
        
        # conv4_x
        self.conv4_x = nn.Sequential(
            BottleneckBlock(in_channels=512, out_channels=1024, is_downsample=True),
            *[BottleneckBlock(in_channels=1024, out_channels=1024) for _ in range(1, 23)]
        )
        
        # conv5_x
        self.conv5_x = nn.Sequential(
            BottleneckBlock(in_channels=1024, out_channels=2048, is_downsample=True),
            *[BottleneckBlock(in_channels=2048, out_channels=2048) for _ in range(1, 3)]
        )
        
        # average pool
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # 1000-d fc
        self.fc = nn.Linear(2048, 1000)

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu') 

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


class ResNet152(nn.Module):
    
    def __init__(self):
        super(ResNet152, self).__init__()
        
        # conv1
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # conv2_x
        self.conv2_x = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            BottleneckBlock(in_channels=64, out_channels=256),
            *[BottleneckBlock(in_channels=256, out_channels=256) for _ in range(1, 3)]
        )
        
        # conv3_x
        self.conv3_x = nn.Sequential(
            BottleneckBlock(in_channels=256, out_channels=512, is_downsample=True),
            *[BottleneckBlock(in_channels=512, out_channels=512) for _ in range(1, 8)]
        )
        
        # conv4_x
        self.conv4_x = nn.Sequential(
            BottleneckBlock(in_channels=512, out_channels=1024, is_downsample=True),
            *[BottleneckBlock(in_channels=1024, out_channels=1024) for _ in range(1, 36)]
        )
        
        # conv5_x
        self.conv5_x = nn.Sequential(
            BottleneckBlock(in_channels=1024, out_channels=2048, is_downsample=True),
            *[BottleneckBlock(in_channels=2048, out_channels=2048) for _ in range(1, 3)]
        )
        
        # average pool
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # 1000-d fc
        self.fc = nn.Linear(2048, 1000)

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu') 

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


def resnet(num_layers):
    if num_layers == 18: return ResNet18()
    if num_layers == 34: return ResNet34()
    if num_layers == 50: return ResNet50()
    if num_layers == 101: return ResNet101()
    if num_layers == 152: return ResNet152()