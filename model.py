from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        out += residual
        out = self.relu(out)
        return out


class ResNetCNN(nn.Module):
    def __init__(self, num_residual_blocks=16, kernel_size=3, padding=1, *args, **kwargs):
        super(ResNetCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(64, 64, kernel_size=kernel_size, padding=padding) for _ in range(num_residual_blocks)]
        )

        self.conv2 = nn.Conv2d(64, 1, kernel_size=3, padding=1)

    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.residual_blocks(out)
        out = self.conv2(out)
        out = residual + out
        return out

class SimpleSRCNN(nn.Module):  # 搭建SRCNN 3层卷积模型，Conve2d（输入层数，输出层数，卷积核大小，步长，填充层）
    def __init__(self, *args, **kwargs):
        super(SimpleSRCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=9, padding=9 // 2)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=5, padding=5 // 2)
        self.conv3 = nn.Conv2d(32, 1, kernel_size=5, padding=5 // 2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x
