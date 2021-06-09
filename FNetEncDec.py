import torch
from torch import nn
from fnet import FNet


class FNetEncoderDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fnet_encoder = FNet(self)