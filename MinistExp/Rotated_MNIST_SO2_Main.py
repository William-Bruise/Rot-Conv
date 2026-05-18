import argparse
import torch
from SteerableCNN_XQ import MinstSteerableCNN
import F_Conv as fn

"""Drop-in script for SO(2)-approx convolution.
Usage mirrors Rotated_MNIST_Main.py but swaps Fconv_PCA -> Fconv_SO2.
"""

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='0')
parser.add_argument('--tranNum', type=int, default=12)
args = parser.parse_args()

# monkey patch keeps existing network definition untouched
fn.Fconv_PCA = fn.Fconv_SO2

model = MinstSteerableCNN(10, args.tranNum)
device = 'cuda:' + args.device if torch.cuda.is_available() else 'cpu'
model = model.to(device)
print('SO2 model instantiated on', device)
print(model.__class__.__name__, 'with Fconv_PCA patched to', fn.Fconv_PCA.__name__)
print('Run the original Rotated_MNIST_Main.py training loop with this patch for direct comparison.')
