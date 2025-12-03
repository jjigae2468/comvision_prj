import torch
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F

resnet50_url = 'https://download.pytorch.org/models/resnet50-19c8e357.pth',

# [추가] SE-Block (Squeeze-and-Excitation) 정의
# 채널 간의 중요도를 학습해서 성능을 높이는 Attention 모듈
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class DetNet(nn.Module):
    # no expansion
    # dilation = 2
    # type B use 1x1 conv
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, block_type='A'):
        super(DetNet, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=2, bias=False, dilation=2)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)
        
        # [추가] DetNet 블록 마지막에 SE-Block 장착!
        self.se = SELayer(self.expansion * planes)

        self.downsample = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes or block_type == 'B':
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        
        # [적용] Attention: 중요한 채널은 살리고 나머지는 억제
        out = self.se(out)

        out += self.downsample(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.in_planes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        # [유지] DetNet 채널은 256으로 유지 (SE-Block이 성능 보완)
        self.layer5 = self._make_detnet_layer(in_channels=2048)
        
        # [추가] Skip Connection을 위한 1x1 Conv (1024 -> 256)
        self.skip_layer = nn.Conv2d(1024, 256, kernel_size=1)

        # [수정] 마지막 Conv 입력 채널 계산
        # DetNet(256) + Skip(256) = 512 채널
        self.conv_end = nn.Conv2d(512, 30, kernel_size=3, stride=1, padding=1, bias=False)
        
        # [유지] 마지막 BN 활성화 (학습 안정성을 위해)
        self.bn_end = nn.BatchNorm2d(30) 

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_planes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        #[3, 4, 6, 3]
        layers = [block(self.in_planes, planes, stride, downsample)]
        self.in_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)

    def _make_detnet_layer(self, in_channels):
        # planes=256 고정 (기존 DetNet 구조 유지)
        layers = [
            DetNet(in_planes=in_channels, planes=256, block_type='B'),
            DetNet(in_planes=256, planes=256, block_type='A'),
            DetNet(in_planes=256, planes=256, block_type='A')
        ]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        
        # [저장] Layer 3 결과 저장 (Skip용)
        x3 = self.layer3(x) 
        
        x = self.layer4(x3)
        x = self.layer5(x) # DetNet 통과 (여기서 SE-Block 동작함)

        # [처리] Skip Connection
        skip = self.skip_layer(x3)       # 1024 -> 256
        skip = F.avg_pool2d(skip, 2, stride=2) # 28x28 -> 14x14
        
        # [결합] (Batch, 256+256, 14, 14)
        x = torch.cat((x, skip), 1)

        # [최종 예측] BN 적용됨
        x = self.conv_end(x)
        x = self.bn_end(x)
        
        x = torch.sigmoid(x)
        x = x.permute(0, 2, 3, 1)

        return x


# resnet50
def resnet50(pretrained=False, **kwargs):
    model_ = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model_.load_state_dict(model_zoo.load_url('https://download.pytorch.org/models/resnet50-19c8e357.pth'))
    return model_


if __name__ == '__main__':
    a = torch.randn((2, 3, 448, 448))
    model = resnet50()
    print(model(a).shape)