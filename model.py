import torch
import torch.nn as nn
import torch.nn.functional as F
from Droppath import DropPath

# auxiliary block
class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


# basic conv block in MobileNetV2
class ConvNormAct(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=(3, 3),
                 stride=(1, 1),
                 padding=0,
                 bias_attr=False,
                 groups=1):
        """
        MobileNetV2 中的基础卷积模块，包括卷积，归一化以及激活
        卷积核权重使用 kaiming normal
        :param in_channels: 输入通道数
        :param out_channels: 输出通道数
        :param kernel_size: 卷积核大小
        :param stride: 步长
        :param padding: padding
        :param bias_attr: 是否使用偏置
        :param groups: groups=inchannels 为 depthwise 卷积
        """
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              groups=groups,
                              bias=bias_attr)
        # self.conv.weight = nn.init.kaiming_normal()
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU()

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = self.act(out)
        return out


class MLP(nn.Module):
    def __init__(self,
                 embed_dim,
                 mlp_ratio,
                 dropout=0.):
        """
        transformer模块中的mlp
        :param embed_dim: 输入维度
        :param mlp_ratio: 隐藏层参数缩放尺度
        :param dropout: 超参
        """
        super().__init__()
        # 权重初始化： truncatedNormal std=.02    bias初始化： constant 0.0   需单独实现
        self.fc1 = nn.Linear(embed_dim,
                             int(embed_dim * mlp_ratio)
                             )
        # 权重初始化： truncatedNormal std=.02    bias初始化： constant 0.0   需单独实现
        self.fc2 = nn.Linear(int(embed_dim * mlp_ratio),
                             embed_dim
                             )
        self.act = nn.SiLU()
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        return x


class Attention(nn.Module):
    def __init__(self,
                 embed_dim,
                 num_heads,
                 qkv_bias=True,
                 dropout=0.,
                 attention_dropout=0.):
        """
        自注意力机制的实现，包括多头自注意力
        :param embed_dim:  embedding dim 也就是qkv的第二维维度大小，类似卷积特征的通道数
        :param num_heads:  多头自注意力头个数
        :param qkv_bias:  qkv是否使用偏执
        :param dropout:  dropout超参
        :param attention_dropout:  超参
        """
        super().__init__()
        self.num_head = num_heads
        # 多头自注意力的实现一般来说不适用扩张维度的方法，而是把embedding dim切分成多个头，每个头计算后cat在一起，
        # 因此多头的个数不会改变参数量和计算量
        self.attn_head_dim = int(embed_dim / self.num_head)
        self.all_head_dim = self.attn_head_dim * self.num_head

        # 权重初始化： truncatedNormal std=.02    bias初始化： constant 0.0   需单独实现
        self.qkv = nn.Linear(embed_dim,
                             self.all_head_dim*3  # weights for q, k, v
                             )
        self.scale = self.attn_head_dim ** -0.5

        # 权重初始化同qkv初始化 输入与输出维度相同
        self.proj = nn.Linear(embed_dim,
                              embed_dim)

        self.attn_dropout = nn.Dropout(attention_dropout)
        self.proj_dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1)

    # 将qkv权重分为多头注意力
    def transpose_multihead(self, x):
        # in shape: [batch_size, P, N, hd]   [批次个数， 图像patch大小， 图像patch个数， embedding_dim]
        B, P, N, d = x.shape
        # 最后一维d就是embedding dim，将embedding dim切分出self.num_head份
        x = x.reshape([B, P, N, self.num_head, -1])
        x = torch.permute(x, (0, 1, 3, 2, 4))
        # out_shape: [batch_size, P, num_heads, N, d]
        return x

    def forward(self, x):
        # x.shape: [B, P, N, d]  [B, 2*2, 256, 96]  [批次个数， 图像patch大小， 图像patch个数， embedding_dim]
        qkv = torch.chunk(self.qkv(x), 3, dim=-1)  # list[[B, 4, 256, 96], [B, 4, 256, 96], [B, 4, 256, 96]]
        # q shape: [B, P, N, d/num_head]
        q, k, v = map(self.transpose_multihead, qkv)  # q.shape [B, 4, 8, 256, 12]

        # q和k的相似度是点乘，计算出的是一个标量，这里用矩阵乘法来一次性计算全部q和k的相似度，因此需要将k转置来计算点积
        attn = torch.matmul(q, k.permute(0, 1, 2, 4, 3))  # q.shape:[B, 4, 8, 256, 12]  k.t.shape:[B, 4, 8, 12, 256]  attn.shape:[B, 4, 8, 256, 256]
        attn = attn * self.scale
        attn = self.softmax(attn)
        attn = self.attn_dropout(attn)  # [batch_size, P, num_heads, N, N]

        z = torch.matmul(attn, v)  # [batch_size, P, num_heads, N, d]
        z = torch.permute(z, (0, 1, 3, 2, 4))
        B, P, N, H, D = z.shape
        z = z.reshape([B, P, N, H * D])
        z = self.proj(z)
        z = self.proj_dropout(z)
        return z


class EncoderLayer(nn.Module):
    def __init__(self,
                 embed_dim,
                 num_heads=8,
                 qkv_bias=True,
                 mlp_ratio=2.0,
                 dropout=0.,
                 attention_dropout=0.,
                 droppath=0.):
        """
        transformer中的基础模块， 包含自注意力，mlp
        :param embed_dim: embedding dim
        :param num_heads: 自注意力头个数
        :param qkv_bias: qkv偏置
        :param mlp_ratio: mlp隐藏层缩放
        :param dropout: 超参
        :param attention_dropout:超参
        :param droppath: 超参
        """
        super().__init__()
        self.att_norm = nn.LayerNorm(embed_dim)
        self.attn = Attention(embed_dim, num_heads, qkv_bias, attention_dropout, dropout)
        self.drop_path = DropPath(droppath) if droppath > 0. else Identity()
        self.mlp_norm = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, mlp_ratio, dropout)

    def forward(self, x):
        h = x
        x = self.att_norm(x)
        x = self.attn(x)
        x = self.drop_path(x)
        x = h + x

        h = x
        x = self.mlp_norm(x)
        x = self.mlp(x)
        x = self.drop_path(x)
        x = x + h
        return x


# Transformer block in MobileVitBlock
class TransformerBlock(nn.Module):
    def __init__(self,
                 embed_dim,
                 num_heads,
                 depth,
                 qkv_bias=True,
                 mlp_ratio=2.0,
                 dropout=0.,
                 attention_dropout=0.,
                 droppath=0.):
        """
        MobileVitBlock 中的标准 Transformer 模块
        Transformer模块一般来说会保持输入输出维度的一致
        :param embed_dim: embedding dim 就是
        :param num_heads: 多头注意力
        :param depth: transformer 模块个数
        :param qkv_bias qkv权重是否使用偏执
        :param mlp_ratio: attention中使用的参数，具体见attention模块
        :param dropout: transformer的dropout
        :param attention_dropout: attention中的dropout
        :param droppath: droppath
        """
        super().__init__()
        # depth_decay = [x.item for x in torch.linspace(0, droppath, depth)]

        layer_list = []
        for i in range(depth):
            layer_list.append(EncoderLayer(embed_dim,
                                           num_heads,
                                           qkv_bias,
                                           mlp_ratio,
                                           dropout,
                                           attention_dropout,
                                           droppath))
        # 不像MobileV2Block中使用Sequential函数是因为每一层使用循环的方式加入的
        self.layers = nn.ModuleList(layer_list)

        # weight初始化为 constant 1.  bias初始化为 constant 0. 需要自己写
        self.norm = nn.LayerNorm(embed_dim,
                                 eps=1e-6)
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        out = self.norm(x)
        return  out


class MobileV2Block(nn.Module):
    def __init__(self, inp, oup, stride=1, expansion=4):
        """
        Mobilenet v2 InvertedResidual block
        :param inp: 输入维度
        :param oup: 输出维度
        :param stride: 步长
        :param expansion: 缩放维度
        """
        super().__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expansion))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expansion != 1:
            # MobileNetV2 中的扩张卷积部分
            layers.append(ConvNormAct(inp, hidden_dim, kernel_size=1))
        layers.extend([
            # dw 当groups参数与输入维度相同时为深度卷积
            ConvNormAct(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim, padding=1),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, (1, 1), (1, 1), 0, bias=False),
            nn.BatchNorm2d(oup),
        ])

        # layers是一个list，*layers表示将layers内的元素拆分成一个个单独的元素送进Sequential，固定写法
        self.conv = nn.Sequential(*layers)
        self.out_channels = oup

    def forward(self, x):
        # 在原版本中stride为2时不使用残差链接进行by pass，只有stride为1时才会用残差链接，主要是为了让维度统一
        if self.use_res_connect:
            return x + self.conv(x)
        return self.conv(x)
    

# MobileViT block in png
class MobileViTBlock(nn.Module):
    def __init__(self,
                 dim,
                 hidden_dim,
                 depth,
                 num_heads=8,
                 qkv_bias=True,
                 mlp_ratio=2.0,
                 dropout=0.,
                 attention_dropout=0.,
                 droppath=0,
                 patch_size=(2, 2)):
        """

        :param dim:输入的卷积层通道数
        :param hidden_dim:local representation 输出的通道数
        :param depth: 标准transformer块的个数
        :param num_heads: 多头注意力中用多少头
        :param qkv_bias:  qkv权重是否使用偏执
        :param mlp_ratio:
        :param dropout: dropout超参数
        :param attention_dropout: 注意力模块中的dropout超参数
        :param droppath: droppath超参数
        :param patch_size: 图像切块像素大小
        """
        super().__init__()
        self.patch_h, self.patch_w = patch_size

        # local representation
        self.conv1 = ConvNormAct(dim, dim, padding=1)
        self.conv2 = ConvNormAct(dim, hidden_dim, kernel_size=1)

        # global representations
        self.transformer = TransformerBlock(embed_dim=hidden_dim,
                                            num_heads=num_heads,
                                            depth=depth,
                                            qkv_bias=qkv_bias,
                                            mlp_ratio=mlp_ratio,
                                            dropout=dropout,
                                            attention_dropout=attention_dropout,
                                            droppath=droppath)

        # fusion
        self.conv3 = ConvNormAct(hidden_dim, dim, kernel_size=1)
        # last conv-n*n, the input is concat of input tensor and conv3 output tensor
        # 输出是有一个残差链接做by pass
        self.conv4 = ConvNormAct(2 * dim, dim, padding=1)

    def forward(self, x):
        # x.shape = [B, C, H, W]
        h = x
        # Local representations
        # x.shape = [B, C, H, W]
        x = self.conv1(x)
        # x.shape = [B, hidden_dim, H, W]
        x = self.conv2(x)
        # Unfold 维度变换 [B, 96, 32, 32] -> [B, 96, 16, 2, 16, 2] -> [B, 96, 2, 2, 16, 16]
        # -> [B, 96, 4, 256] -> [B, 4, 256, 96]
        B, C, H, W = x.shape
        # Unfold
        x = x.reshape([B, C, H//self.patch_h, self.patch_h, W//self.patch_w, self.patch_w])
        # x.shape = [B, hidden_dim, H//patch_h, patch_h, W//patch_w, patch_w]
        x = torch.permute(x, (0, 1, 3, 5, 2, 4))
        # x.shape = [B, hidden_dim, patch_h, patch_w, H//patch_h, W//patch_w]
        x = x.reshape([B, C, (self.patch_h * self.patch_w), -1])
        # x.shape = [B, ws**2, n_windows**2, hidden_dim]
        x = torch.permute(x, (0, 2, 3, 1))

        # L * Trasformer [B, 4, 256, 96] -> [B, 4, 256, 96]
        x = self.transformer(x)

        # Fold
        x = x.reshape([B, self.patch_h, self.patch_w, H//self.patch_h, W//self.patch_w, C])
        x = torch.permute(x, (0, 5, 3, 1, 4, 2))
        x = x.reshape([B, C, H, W])

        # Fusion
        x = self.conv3(x)
        x = torch.cat((h, x), dim=1)
        x= self.conv4(x)
        return x


# 不使用patch embedding， 使用3d卷积，刚好应对原论文中的四次卷积，把空间尺度降下去，然后进入transformer，不改变维度，直接输入到后续
class MobileViTBlock3d(nn.Module):
    def __init__(self,
                 image_size=(224, 224, 155),
                 dim=1,
                 hidden_dim=16,
                 depth=2,
                 num_heads=8,
                 qkv_bias=True,
                 mlp_ratio=2.0,
                 dropout=0.,
                 attention_dropout=0.,
                 droppath=0,
                 patch_size=(2, 2, 2)):
        """

        :param dim:输入的卷积层通道数
        :param hidden_dim:local representation 输出的通道数
        :param depth: 标准transformer块的个数
        :param num_heads: 多头注意力中用多少头
        :param qkv_bias:  qkv权重是否使用偏执
        :param mlp_ratio:
        :param dropout: dropout超参数
        :param attention_dropout: 注意力模块中的dropout超参数
        :param droppath: droppath超参数
        :param patch_size: 图像切块像素大小
        """
        super().__init__()
        self.patch_h, self.patch_w, self.patch_z = patch_size

        # local representation
        self.conv1 = ConvNormAct(dim, dim, padding=1)
        self.conv2 = ConvNormAct(dim, hidden_dim, kernel_size=1)

        # global representations
        self.transformer = TransformerBlock(embed_dim=96,
                                            num_heads=num_heads,
                                            depth=depth,
                                            qkv_bias=qkv_bias,
                                            mlp_ratio=mlp_ratio,
                                            dropout=dropout,
                                            attention_dropout=attention_dropout,
                                            droppath=droppath)

        # fusion
        self.conv3 = ConvNormAct(hidden_dim, dim, kernel_size=1)
        # last conv-n*n, the input is concat of input tensor and conv3 output tensor
        # 输出是有一个残差链接做by pass
        self.conv4 = ConvNormAct(2 * dim, dim, padding=1)


    def forward(self, x):
        # x.shape = [B, C, H, W, Z]
        h = x
        # Local representations
        # x.shape = [B, C, H, W]
        x = self.conv1(x)  # 替换成3d的convnormact
        # x.shape = [B, hidden_dim, H, W]
        x = self.conv2(x)
        # Unfold 维度变换 [B, 96, 32, 32， 32] -> [B, 96, 16, 2, 16, 2] -> [B, 96, 2, 2, 16, 16]
        # -> [B, 96, 4, 256] -> [B, 4, 256, 96]
        B, C, H, W, Z = x.shape
        # Unfold
        x = x.reshape([B, C, H//self.patch_h, self.patch_h, W//self.patch_w, self.patch_w, Z//self.patch_z, self.patch_z])
        # [B, 96, H//2, 2, W//2, 2, Z//2, 2]
        x = torch.permute(x, (0, 1, 3, 5, 7, 2, 4, 6))
        # [B, 96, 2, 2, 2, H//2, W//2, Z//2]
        x = x.reshape([B, C, (self.patch_h * self.patch_w * self.patch_z), -1])
        # x.shape = [B, 96, 2*2*2, H//2 * W//2 * Z//2]
        x = torch.permute(x, (0, 2, 3, 1))
        # x.shape = [B,2*2*2, H//2 * W//2 * Z//2, 96]

        # L * Trasformer [B, 4, 256, 96] -> [B, 4, 256, 96]
        x = self.transformer(x)

        # Fold
        x = x.reshape([B, self.patch_h, self.patch_w, self.patch_z, H//self.patch_h, W//self.patch_w, Z//self.patch_z, C])
        x = torch.permute(x, (0, 7, 4, 1, 5, 2, 6, 3))
        x = x.reshape([B, C, H, W, Z])

        # Fusion
        x = self.conv3(x)
        x = torch.cat((h, x), dim=1)
        x= self.conv4(x)
        return x


# integrate all ingredients
class MobileViT(nn.Module):
    def __init__(self,
                 in_channels=3,
                 dims=[16, 32, 48, 48, 48, 64, 80, 96, 384],  # XS
                 hidden_dims=[96, 120, 144],  # d: hidden dims in mobilevit block
                 num_classes=1000):
        """
        MobileVit
        :param in_channels:
        :param dims:
        :param hidden_dims:
        :param num_classes: 用于分类，用于分割不需要这一项并且需要修改输出
        """
        super().__init__()
        # [B, 3, 256, 256]
        self.conv3x3 = ConvNormAct(in_channels, dims[0], kernel_size=3, stride=2, padding=1)
        # [B, 16, 128, 128] self.conv3*3输出，下同理
        self.mv2_block_1 = MobileV2Block(dims[0], dims[1])

        # [B, 32, 128, 128]
        self.mv2_block_2 = MobileV2Block(dims[1], dims[2], stride=2)
        # [B, 48, 64, 64]
        self.mv2_block_3 = MobileV2Block(dims[2], dims[3])
        # [B, 48, 64, 64]
        self.mv2_block_4 = MobileV2Block(dims[3], dims[4])  # repeat = 2

        # [B, 48, 64, 64]
        self.mv2_block_5 = MobileV2Block(dims[4], dims[5], stride=2)
        # [B, 64, 32, 32]
        self.mvit_block_1 = MobileViTBlock(dims[5], hidden_dims[0], depth=2)

        # [B, 64, 32, 32]
        self.mv2_block_6 = MobileV2Block(dims[5], dims[6], stride=2)
        # [B, 80, 16, 16]
        self.mvit_block_2 = MobileViTBlock(dims[6], hidden_dims[1], depth=4)

        # [B, 80, 16, 16]
        self.mv2_block_7 = MobileV2Block(dims[6], dims[7], stride=2)
        # [B, 96, 8, 8]
        self.mvit_block_3 = MobileViTBlock(dims[7], hidden_dims[2], depth=3)

        # [B, 96, 8, 8]
        self.conv1x1 = ConvNormAct(dims[7], dims[8], kernel_size=1)

        # [B, 384, 8, 8]
        self.pool = nn.AdaptiveAvgPool2d(1)
        # [B, 384, 1, 1]
        # 输出分类结果，分割需要换成自己的分割头
        self.linear = nn.Linear(dims[8], num_classes)
        # [B, 1000]

    def forward(self, x):
        x = self.conv3x3(x)
        x = self.mv2_block_1(x)

        x = self.mv2_block_2(x)
        x = self.mv2_block_3(x)
        x = self.mv2_block_4(x)

        x = self.mv2_block_5(x)
        x = self.mvit_block_1(x)

        x = self.mv2_block_6(x)
        x = self.mvit_block_2(x)

        x = self.mv2_block_7(x)
        x = self.mvit_block_3(x)
        x = self.conv1x1(x)

        x = self.pool(x)
        x = x.reshape(x.shape[:2])
        x = self.linear(x)

        return x


def build_mobile_vit(config):
    """Build MobileViT by reading options in config object
    Args:
        config: config instance contains setting options
    Returns:
        model: MobileViT model
    """
    model = MobileViT(in_channels=config.MODEL.IN_CHANNELS,
                      dims=config.MODEL.DIMS,  # XS: [16, 32, 48, 48, 48, 64, 80, 96, 384]
                      hidden_dims=config.MODEL.HIDDEN_DIMS, # XS: [96, 120, 144], # d: hidden dims in mobilevit block
                      num_classes=config.MODEL.NUM_CLASSES)
    return model


if __name__ == "__main__":
    a = torch.arange(1*4*256*96, dtype=torch.float32).view(1, 4, 256, 96)
    b = TransformerBlock(96, depth=1, num_heads=8)
    out = b(a)

    print(out.shape)
