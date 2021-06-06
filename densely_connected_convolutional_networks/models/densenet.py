import torch
import torch.nn as nn


__all__ = ['densenet']


def conv1x1(in_channels, out_channels, padding=0, stride=1, bias=False):
    return nn.Sequential(nn.BatchNorm2d(in_channels),
                         nn.ReLU(inplace=True),
                         nn.Conv2d(in_channels, out_channels,
                                   kernel_size=1, stride=stride,
                                   padding=padding, bias=bias))


def conv3x3(in_channels, out_channels, padding=0, stride=1, bias=False):
    return nn.Sequential(nn.BatchNorm2d(in_channels),
                         nn.ReLU(inplace=True),
                         nn.Conv2d(in_channels, out_channels,
                                   kernel_size=3, stride=stride,
                                   padding=padding, bias=bias))


class DenseBlock(nn.Module):
    
    def __init__(self, k0, k, n_layers):
        super(DenseBlock, self).__init__()
        
        self.k0 = k0
        self.k = k
        self.n_layers = n_layers
        self.layers = nn.Sequential(
            *[self._make_bottleneck_layer(l) for l in range(1, n_layers + 1)]
        )
    
    def forward(self, x):
        inputs = [x]
        for i in range(self.n_layers):
            inputs.append(self.layers[i](torch.cat(inputs, 1)))
            
        return torch.cat(inputs, 1)
    
    def _make_bottleneck_layer(self, l):
        in_channels = self.k0 + self.k * (l - 1)
        out_channels = self.k
        
        return nn.Sequential(conv1x1(in_channels=in_channels,
                                     out_channels=4*self.k),
                             conv3x3(in_channels=4*self.k,
                                     out_channels=out_channels, padding=1))


class TransitionLayer(nn.Module):
    
    def __init__(self, theta, m):
        super(TransitionLayer, self).__init__()
        
        self.conv = conv1x1(m, int(theta * m))
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        
        return x


class DenseNet121(nn.Module):
    
    def __init__(self):
        super(DenseNet121, self).__init__()
        
        k = 32        
        dict_k0 = {"block_1": 64, "block_2": 128, "block_3": 256, "block_4": 512}
        dict_n_layers = {"block_1": 6, "block_2": 12, "block_3": 24, "block_4": 16}
        dict_m = {"layer_1": 256, "layer_2": 512, "layer_3": 1024}
        
        # Convolution
        self.conv = nn.Sequential(
            nn.Conv2d(3, 2 * k, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(2 * k),
            nn.ReLU(inplace=True)
        )
        
        # Pooling
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Dense Block 1 (output channels = k0 + k * l = 64 + 32 * 6 = 256)
        self.dense_block1 = DenseBlock(dict_k0["block_1"], k, dict_n_layers["block_1"])
        
        # Transition Layer 1 (output channels = m * theta = 256 * 0.5 = 128)
        self.trans_layer1 = TransitionLayer(0.5, dict_m["layer_1"])
        
        # Dense Block 2 (output channels = k0 + k * l = 128 + 32 * 12 = 512)
        self.dense_block2 = DenseBlock(dict_k0["block_2"], k, dict_n_layers["block_2"])
        # output channels = k0 + k * l = 512
        
        # Transition Layer 2 (output channels = m * theta = 512 * 0.5 = 256)
        self.trans_layer2 = TransitionLayer(0.5, dict_m["layer_2"])
        
        # Dense Block 3 (output channels = k0 + k * l = 256 + 32 * 24 = 1024)
        self.dense_block3 = DenseBlock(dict_k0["block_3"], k, dict_n_layers["block_3"])
        
        # Transition Layer 3 (output channels = m * theta = 1024 * 0.5 = 512)
        self.trans_layer3 = TransitionLayer(0.5, dict_m["layer_3"])
        
        # Dense Block 4 (output channels = k0 + k * l = 512 + 32 * 16 = 1024)
        self.dense_block4 = DenseBlock(dict_k0["block_4"], k, dict_n_layers["block_4"])
        
        self.bn = nn.BatchNorm2d(1024)
        
        # Classification Layer
        self.gap = nn.AvgPool2d(kernel_size=7)
        self.fc = nn.Linear(1024, 1000)

        # Weight Initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        
        x = self.dense_block1(x)
        x = self.trans_layer1(x)
        
        x = self.dense_block2(x)
        x = self.trans_layer2(x)
        
        x = self.dense_block3(x)
        x = self.trans_layer3(x)
        
        x = self.dense_block4(x)
        x = self.bn(x)
        
        x = self.gap(x)
        x = x.view(-1, 1024)
        x = self.fc(x)
        
        return x


class DenseNet169(nn.Module):
    
    def __init__(self):
        super(DenseNet169, self).__init__()
        
        k = 32        
        dict_k0 = {"block_1": 64, "block_2": 128, "block_3": 256, "block_4": 640}
        dict_n_layers = {"block_1": 6, "block_2": 12, "block_3": 32, "block_4": 32}
        dict_m = {"layer_1": 256, "layer_2": 512, "layer_3": 1280}
        
        # Convolution
        self.conv = nn.Sequential(
            nn.Conv2d(3, 2 * k, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(2 * k),
            nn.ReLU(inplace=True)
        )
        
        # Pooling
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Dense Block 1 (output channels = k0 + k * l = 64 + 32 * 6 = 256)
        self.dense_block1 = DenseBlock(dict_k0["block_1"], k, dict_n_layers["block_1"])
        
        # Transition Layer 1 (output channels = m * theta = 256 * 0.5 = 128)
        self.trans_layer1 = TransitionLayer(0.5, dict_m["layer_1"])
        
        # Dense Block 2 (output channels = k0 + k * l = 128 + 32 * 12 = 512)
        self.dense_block2 = DenseBlock(dict_k0["block_2"], k, dict_n_layers["block_2"])
        
        # Transition Layer 2 (output channels = m * theta = 512 * 0.5 = 256)
        self.trans_layer2 = TransitionLayer(0.5, dict_m["layer_2"])
        
        # Dense Block 3 (output channels = k0 + k * l = 256 + 32 * 32 = 1280)
        self.dense_block3 = DenseBlock(dict_k0["block_3"], k, dict_n_layers["block_3"])
        
        # Transition Layer 2 (output channels = m * theta = 1280 * 0.5 = 640)
        self.trans_layer3 = TransitionLayer(0.5, dict_m["layer_3"])
        
        # Dense Block 4 (output channels = k0 + k * l = 640 + 32 * 32 = 1664)
        self.dense_block4 = DenseBlock(dict_k0["block_4"], k, dict_n_layers["block_4"])
        
        self.bn = nn.BatchNorm2d(1664)
        
        # Classification Layer
        self.gap = nn.AvgPool2d(kernel_size=7)
        self.fc = nn.Linear(1664, 1000)

        # Weight Initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        
        x = self.dense_block1(x)
        x = self.trans_layer1(x)
        
        x = self.dense_block2(x)
        x = self.trans_layer2(x)
        
        x = self.dense_block3(x)
        x = self.trans_layer3(x)
        
        x = self.dense_block4(x)
        x = self.bn(x)
        
        x = self.gap(x)
        x = x.view(-1, 1664)
        x = self.fc(x)
        
        return x


class DenseNet201(nn.Module):
    
    def __init__(self):
        super(DenseNet201, self).__init__()
        
        k = 32        
        dict_k0 = {"block_1": 64, "block_2": 128, "block_3": 256, "block_4": 896}
        dict_n_layers = {"block_1": 6, "block_2": 12, "block_3": 48, "block_4": 32}
        dict_m = {"layer_1": 256, "layer_2": 512, "layer_3": 1792}
        
        # Convolution
        self.conv = nn.Sequential(
            nn.Conv2d(3, 2 * k, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(2 * k),
            nn.ReLU(inplace=True)
        )
        
        # Pooling
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Dense Block 1 (output channels = k0 + k * l = 64 + 32 * 6 = 256)
        self.dense_block1 = DenseBlock(dict_k0["block_1"], k, dict_n_layers["block_1"])
        
        # Transition Layer 1 (output channels = m * theta = 256 * 0.5 = 128)
        self.trans_layer1 = TransitionLayer(0.5, dict_m["layer_1"])
        
        # Dense Block 2 (output channels = k0 + k * l = 128 + 32 * 12 = 512)
        self.dense_block2 = DenseBlock(dict_k0["block_2"], k, dict_n_layers["block_2"])
        
        # Transition Layer 2 (output channels = m * theta = 512 * 0.5 = 256)
        self.trans_layer2 = TransitionLayer(0.5, dict_m["layer_2"])
        
        # Dense Block 3 (output channels = k0 + k * l = 256 + 32 * 48 = 1792)
        self.dense_block3 = DenseBlock(dict_k0["block_3"], k, dict_n_layers["block_3"])
        
        # Transition Layer 3 (output channels = m * theta = 1792 * 0.5 = 896)
        self.trans_layer3 = TransitionLayer(0.5, dict_m["layer_3"])
        
        # Dense Block 4 (output channels = k0 + k * l = 896 + 32 * 32 = 1920)
        self.dense_block4 = DenseBlock(dict_k0["block_4"], k, dict_n_layers["block_4"])
        
        self.bn = nn.BatchNorm2d(1920)
        
        # Classification Layer
        self.gap = nn.AvgPool2d(kernel_size=7)
        self.fc = nn.Linear(1920, 1000)

        # Weight Initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        
        x = self.dense_block1(x)
        x = self.trans_layer1(x)
        
        x = self.dense_block2(x)
        x = self.trans_layer2(x)
        
        x = self.dense_block3(x)
        x = self.trans_layer3(x)
        
        x = self.dense_block4(x)
        x = self.bn(x)
        
        x = self.gap(x)
        x = x.view(-1, 1920)
        x = self.fc(x)
        
        return x


class DenseNet264(nn.Module):
    
    def __init__(self):
        super(DenseNet264, self).__init__()
        
        k = 32        
        dict_k0 = {"block_1": 64, "block_2": 128, "block_3": 256, "block_4": 1152}
        dict_n_layers = {"block_1": 6, "block_2": 12, "block_3": 64, "block_4": 48}
        dict_m = {"layer_1": 256, "layer_2": 512, "layer_3": 2304}
        
        # Convolution
        self.conv = nn.Sequential(
            nn.Conv2d(3, 2 * k, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(2 * k),
            nn.ReLU(inplace=True)
        )
        
        # Pooling
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Dense Block 1 (output channels = k0 + k * l = 64 + 32 * 6 = 256)
        self.dense_block1 = DenseBlock(dict_k0["block_1"], k, dict_n_layers["block_1"])
        
        # Transition Layer 1 (output channels = m * theta = 256 * 0.5 = 128)
        self.trans_layer1 = TransitionLayer(0.5, dict_m["layer_1"])
        
        # Dense Block 2 (output channels = k0 + k * l = 128 + 32 * 12 = 512)
        self.dense_block2 = DenseBlock(dict_k0["block_2"], k, dict_n_layers["block_2"])
        
        # Transition Layer 2 (output channels = m * theta = 512 * 0.5 = 256)
        self.trans_layer2 = TransitionLayer(0.5, dict_m["layer_2"])
        
        # Dense Block 3 (output channels = k0 + k * l = 256 + 32 * 64 = 2304)
        self.dense_block3 = DenseBlock(dict_k0["block_3"], k, dict_n_layers["block_3"])
        
        # Transition Layer 3 (output channels = m * theta = 2304 * 0.5 = 1152)
        self.trans_layer3 = TransitionLayer(0.5, dict_m["layer_3"])
        
        # Dense Block 4 (output channels = k0 + k * l = 1152 + 32 * 48 = 2688)
        self.dense_block4 = DenseBlock(dict_k0["block_4"], k, dict_n_layers["block_4"])
        
        self.bn = nn.BatchNorm2d(2688)
        
        # Classification Layer
        self.gap = nn.AvgPool2d(kernel_size=7)
        self.fc = nn.Linear(2688, 1000)

        # Weight Initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        
        x = self.dense_block1(x)
        x = self.trans_layer1(x)
        
        x = self.dense_block2(x)
        x = self.trans_layer2(x)
        
        x = self.dense_block3(x)
        x = self.trans_layer3(x)
        
        x = self.dense_block4(x)
        x = self.bn(x)
        
        x = self.gap(x)
        x = x.view(-1, 2688)
        x = self.fc(x)
        
        return x


def densenet(num_layers):
    if num_layers == 121: return DenseNet121()
    if num_layers == 169: return DenseNet169()
    if num_layers == 201: return DenseNet201()
    if num_layers == 264: return DenseNet264()