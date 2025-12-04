import torch
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F

resnet50_url = 'https://download.pytorch.org/models/resnet50-19c8e357.pth',


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
        
        # [수정] DetNet 채널 확장 (256 -> 512)
        # Layer4의 출력 채널은 2048, 이를 512로 변환
        self.layer5 = self._make_detnet_layer(in_channels=2048, planes=512)
        
        # [추가] Skip Connection을 위한 1x1 Conv
        # Layer3의 출력(1024채널)을 가져와서 256채널로 압축
        self.skip_layer = nn.Conv2d(1024, 256, kernel_size=1)

        # [수정] 마지막 Conv 입력 채널 변경
        # DetNet 출력(512) + Skip 출력(256) = 768 채널
        # *** 중요: BN을 제거했으므로 bias=True로 설정하여 편향 학습을 가능하게 함 ***
        self.conv_end = nn.Conv2d(768, 30, kernel_size=3, stride=1, padding=1, bias=True)
        
        # [삭제] 마지막 BN은 회귀 문제(좌표 예측)를 방해하므로 제거
        # self.bn_end = nn.BatchNorm2d(30) 

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

    def _make_detnet_layer(self, in_channels, planes):
        # [수정] planes 인자를 받아서 채널 수를 유동적으로 조절 가능하게 변경
        layers = [
            DetNet(in_planes=in_channels, planes=planes, block_type='B'),
            DetNet(in_planes=planes, planes=planes, block_type='A'),
            DetNet(in_planes=planes, planes=planes, block_type='A')
        ]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        
        # [변경] Layer 3에서 Skip Connection 분기
        x3 = self.layer3(x) 
        
        # 메인 경로는 계속 진행
        x = self.layer4(x3)
        x = self.layer5(x) # DetNet 통과 (14x14, 512ch)

        # [추가] Skip Connection 처리
        # 1. 채널 압축 (1024 -> 256)
        skip = self.skip_layer(x3) 
        # 2. 크기 맞춤 (28x28 -> 14x14)
        skip = F.avg_pool2d(skip, 2, stride=2) 

        # 3. 채널 방향 결합 (Concat)
        # 결과: (Batch, 768, 14, 14)
        x = torch.cat((x, skip), 1)

        # 최종 예측 (BN 제거됨, Conv에는 Bias 포함됨)
        x = self.conv_end(x)
        # x = self.bn_end(x) # 사용 안 함
        
        x = torch.sigmoid(x)
        x = x.permute(0, 2, 3, 1)  # (-1, 14, 14, 30)

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