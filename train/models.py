import torch
import torch.nn as nn


cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, vgg_name):
        super(VGG, self).__init__()
        self.layers_num = 0
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Sequential(
            nn.Linear(512, 10)
        )

    def forward(self, x, startLayer, endLayer, isTrain):
        if isTrain:
            x = self.features(x)
            x = x.view(x.size(0), -1)
            x = self.classifier(x)
        else:
            if startLayer == endLayer:
                if startLayer == self.layers_num:
                    x = x.view(x.size(0), -1)
                    x = self.classifier[startLayer - self.layers_num](x)
                elif startLayer < self.layers_num:
                    x = self.features[startLayer](x)
                else:
                    x = self.classifier[startLayer - self.layers_num](x)
            else:
                for i in range(startLayer, endLayer + 1):
                    if i < self.layers_num:
                        x = self.features[i](x)
                    elif i == self.layers_num:
                        x = x.view(x.size(0), -1)
                        x = self.classifier[i - self.layers_num](x)
                    else:
                        x = self.classifier[i - self.layers_num](x)
        return x

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
                self.layers_num += 1
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
                self.layers_num += 3
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        self.layers_num += 1
        print(f"number of layers: {self.layers_num}")
        return nn.Sequential(*layers)


def VGG11():
    return VGG('VGG11')


def VGG13():
    return VGG('VGG13')


def VGG16():
    return VGG('VGG16')


def VGG19():
    return VGG('VGG19')


class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(1024, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 10),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 1024)
        x = self.classifier(x)
        return x

    def forward(self, x, startLayer, endLayer, isTrain):
        if isTrain:
            x = self.features(x)
            x = x.view(x.size(0), 1024)
            x = self.classifier(x)
        else:
            if startLayer == endLayer:
                if startLayer == 13:
                    x = x.view(x.size(0), 1024)
                    x = self.classifier[startLayer-13](x)
                elif startLayer < 13:
                    x = self.features[startLayer](x)
                else:
                    x = self.classifier[startLayer-13](x)
            else:
                for i in range(startLayer, endLayer + 1):
                    if i < 13:
                        x = self.features[i](x)
                    elif i == 13:
                        x = x.view(x.size(0), 1024)
                        x = self.classifier[i-13](x)
                    else:
                        x = self.classifier[i-13](x)
        return x
