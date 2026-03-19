import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True):
        super(SeparableConv2d, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


def constraint(max_val, param, eps=1e-8):
    norm = param.norm(2, dim=0, keepdim=True)
    desired = torch.clamp(norm, 0, max_val)
    param = param * (desired / (eps + norm))


class EEGNet(nn.Module):
    def __init__(self, F1, D, C, T, dropout, N, filter_len=64, hyper_sphere=False):
        super(EEGNet, self).__init__()
        f_s = [(1, filter_len), (C, 1), (1, 4), (1, 16), (1, 8)]
        F2 = D * F1
        l = int(np.floor((f_s[0][1] - 1) / 2))
        r = int(np.ceil((f_s[0][1] - 1) / 2))

        l2 = int(np.floor((f_s[3][1] - 1) / 2))
        r2 = int(np.ceil((f_s[3][1] - 1) / 2))

        self.block1 = nn.Sequential(
            nn.ConstantPad1d((l, r), 0),
            nn.Conv2d(1, F1, f_s[0], bias=False),  # time conv
            nn.BatchNorm2d(F1),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(F1, D * F1, f_s[1], groups=F1, bias=False),  # depthwise_conv spatial conv
            nn.BatchNorm2d(D * F1),
            nn.ELU(),
            nn.AvgPool2d(f_s[2], stride=(1, 4)),
            nn.Dropout(dropout),
        )
        self.block3 = nn.Sequential(
            nn.ConstantPad1d((l2, r2), 0),
            SeparableConv2d(D * F1, F2, f_s[3], bias=False),  # time conv
            nn.BatchNorm2d(F2),
            nn.AvgPool2d(f_s[4], stride=(1, 8)),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(F2 * (T // 32), N),
        )

        self.hyper_sphere = hyper_sphere

    def extract(self, x):
        x = x.unsqueeze(1)
        self.max_norm()
        x = self.block3(self.block2(self.block1(x)))

        if self.hyper_sphere:
            '''
            project features to hyper sphere, 
            this will make those features more easy to classified
            by linear classifier. In this way i could also use linear
            svm to classify and svm ensembles.
            '''
            x = F.normalize(x, p=2, dim=1)  # 2 is hyper sphere

        return x

    def forward(self, x):
        features = self.extract(x)
        classifiers = self.fc(features)

        return classifiers

    def max_norm(self):
        for name, param in self.named_parameters():
            if 'bias' not in name:  # bias项不进行最大范数约束，只约束weight项
                if (name == 'block2.0.weight'):  # depthwise conv
                    constraint(1, param)
                elif (name == 'fc.1.weight'):
                    constraint(0.25, param)


if __name__ == '__main__':
    from torchsummary import summary

    eegnet = EEGNet(F1=8, D=2, C=32, T=600, dropout=0.25, N=2, filter_len=64)
    summary(eegnet, input_size=(32, 600), device='cpu')
