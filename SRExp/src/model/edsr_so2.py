from model import common
import torch.nn as nn
from model import F_Conv as fn

def make_model(args, parent=False):
    return EDSR_SO2(args)

class EDSR_SO2(nn.Module):
    def __init__(self, args):
        super(EDSR_SO2, self).__init__()
        n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        kernel_size = int(args.kernel_size)
        iniScale = args.ini_scale
        scale = args.scale[0]
        act = nn.ReLU(True)
        inP = kernel_size
        tranNum = args.tranNum
        self.sub_mean = common.MeanShift(args.rgb_range)
        self.add_mean = common.MeanShift(args.rgb_range, sign=1)
        Smooth = False

        m_head = [fn.Fconv_SO2(kernel_size, args.n_colors, n_feats, tranNum, inP=inP, padding=(kernel_size-1)//2, ifIni=1, Smooth=Smooth, iniScale=iniScale)]
        m_body = [
            fn.ResBlock(fn.Fconv_SO2, n_feats, kernel_size, tranNum=tranNum, inP=inP, act=act, res_scale=args.res_scale, Smooth=Smooth, iniScale=iniScale)
            for _ in range(n_resblocks)
        ]
        conv = common.default_conv
        n_feats = n_feats * tranNum
        m_tail = [common.Upsampler(conv, scale, n_feats, act=False), conv(n_feats, args.n_colors, 3)]

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        x = self.sub_mean(x)
        x = self.head(x)
        res = self.body(x)
        res += x
        x = self.tail(res)
        x = self.add_mean(x)
        return x
