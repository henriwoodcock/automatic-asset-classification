import torch
from torch import nn
from fastai.torch_core import *
from fastai.core import ifnone
from fastai.layers import *

class reshape(nn.Module):
    '''a torch layer to reshape the input into size = shape = type list'''
    def __init__(self, shape):
        super(reshape, self).__init__()
        self.shape = shape
    def forward(self, x): return x.reshape(self.shape)

class convblock(nn.Module):
    '''
    a convolutional block used in the model:
    conv(in, out) -> batchnorm(out) -> relu
    '''
    def __init__(self, in_:int, out:int):
      super().__init__()
      self.conv1 = nn.Conv2d(in_, out, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      self.bn = nn.BatchNorm2d(out, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
      x = self.conv1(x)
      x = self.bn(x)
      x = self.relu(x)
      return x

class downsamp(nn.Module):
    ''' a downsampling block. using adaptive max pooling so select the size to be outputted and the scale you would like output ie out (3,10,10) is a downsamp(3, 10).
    '''
    def __init__(self, size:int, scale:int=2):
      super().__init__()
      self.pool = nn.AdaptiveMaxPool2d(scale)
      self.bn = nn.BatchNorm2d(size, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      self.relu = nn.ReLU(inplace = True)

    def forward(self,x):
      x = self.pool(x)
      x = self.bn(x)
      x = self.relu(x)
      return x

class Upsample(nn.Module):
    '''
    upsample by scale = scale. Ins and outs are input. Upsampling method is nearest neighbour.
    '''
    def __init__(self, in_:int, out:int, scale:int=2):
      super().__init__()
      self.upsample = nn.Upsample(scale_factor=scale, mode='nearest')
      self.bn = nn.BatchNorm2d(in_, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      self.conv1 = nn.Sequential(
              nn.Conv2d(in_, out, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
              nn.BatchNorm2d(out, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
              nn.ReLU(inplace=True)
              )

    def forward(self, x):
      x = self.upsample(x)
      x = self.bn(x)
      x = self.conv1(x)
      return x

def icnr(x, scale=2, init=nn.init.kaiming_normal_):
    '''ICNR init of `x`, with `scale` and `init` function.'''
    ni,nf,h,w = x.shape
    ni2 = int(ni/(scale**2))
    k = init(torch.zeros([ni2,nf,h,w])).transpose(0, 1)
    k = k.contiguous().view(ni2, nf, -1)
    k = k.repeat(1, 1, scale**2)
    k = k.contiguous().view([nf,ni,h,w]).transpose(0, 1)
    x.data.copy_(k)

class PixelShuffle_ICNR(nn.Module):
    '''Upsample by `scale` from `ni` filters to `nf` (default `ni`), using `nn.PixelShuffle`, `icnr` init, and `weight_norm`.'''
    def __init__(self, ni:int, nf:int=None, scale:int=2, blur:bool=False, norm_type=NormType.Weight):
        super().__init__()
        nf = ifnone(nf, ni)
        self.conv = conv_layer(ni, nf*(scale**2), ks=1, norm_type=norm_type, use_activ=False)
        icnr(self.conv[0].weight)
        self.shuf = nn.PixelShuffle(scale)
        # Blurring over (h*w) kernel
        # "Super-Resolution using Convolutional Neural Networks without Any Checkerboard Artifacts"
        # - https://arxiv.org/abs/1806.02658
        self.pad = nn.ReplicationPad2d((1,0,1,0))
        self.blur = nn.AvgPool2d(2, stride=1)
        self.relu = nn.ReLU(inplace = True)

    def forward(self,x):
        x = self.shuf(self.relu(self.conv(x)))
        return self.blur(self.pad(x)) if self.blur else x

class AdaptiveConcatPool2d(Module):
    "Layer that concats `AdaptiveAvgPool2d` and `AdaptiveMaxPool2d`."
    def __init__(self, sz:Optional[int]=None):
        "Output will be 2*sz or 2 if sz is None"
        self.output_size = sz or 1
        self.ap = nn.AdaptiveAvgPool2d(self.output_size)
        self.mp = nn.AdaptiveMaxPool2d(self.output_size)

    def forward(self, x): return torch.cat([self.mp(x), self.ap(x)], 1)
