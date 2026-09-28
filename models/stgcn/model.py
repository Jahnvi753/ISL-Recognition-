import torch
import torch.nn as nn

from .graph import get_graph


class STGCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, A, stride=1):
        super().__init__()

        self.register_buffer(
            "A",
            torch.tensor(A, dtype=torch.float32)
        )

        self.gcn = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1
        )

        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=(9, 1),
                padding=(4, 0),
                stride=(stride, 1)
            ),
            nn.BatchNorm2d(out_channels)
        )

        if in_channels == out_channels and stride == 1:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=(stride, 1)
                ),
                nn.BatchNorm2d(out_channels)
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = self.residual(x)

        x = torch.einsum("nctv,vw->nctw", x, self.A)
        x = self.gcn(x)
        x = self.tcn(x)

        x = x + residual

        return self.relu(x)


class STGCN(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()

        A = get_graph()

        self.data_bn = nn.BatchNorm1d(75 * 3)

        self.block1 = STGCNBlock(
            in_channels=3,
            out_channels=64,
            A=A
        )

        self.block2 = STGCNBlock(
            in_channels=64,
            out_channels=128,
            A=A
        )

        self.block3 = STGCNBlock(
            in_channels=128,
            out_channels=256,
            A=A
        )

        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        # Input: (N, T, 75, 3)

        N, T, V, C = x.shape

        x = x.view(N, T, V * C)

        x = self.data_bn(x.transpose(1, 2))

        x = x.transpose(1, 2)

        x = x.view(N, T, V, C)

        x = x.permute(0, 3, 1, 2).contiguous()

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        x = x.mean(dim=(2, 3))

        return self.classifier(x)