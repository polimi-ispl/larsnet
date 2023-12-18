import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
from torch import Tensor


class UNetUtils:
    """Utility class for the UNet models"""

    def __init__(self,
                 F: int = None,
                 T: int = None,
                 n_fft: int = 4096,
                 win_length: int = None,
                 hop_length: int = None,
                 center: bool = True,
                 device='cpu',
                 ):
        """
        :param F: Number of frequency bins of the Magnitude STFT that are kept before feeding it to the UNet.
        :param T: Number of time frames of each Magnitude STFT sample that must be fed to the UNet.
        :param n_fft: Length of the windowed signal after padding with zeros and before the STFT.
        :param win_length: Length of the STFT window.
        :param hop_length: Stride of the STFT window.
        :param center: Whether to pad the input on both sides so that each time frame is centered.
        :param device: Target device, default to 'cpu'
        """

        self.n_fft = n_fft
        self.win_length = n_fft if win_length is None else win_length
        self.hop_length = self.win_length // 4 if hop_length is None else hop_length
        self.hann_window = torch.hann_window(self.win_length, periodic=True).to(device)
        self.center = center
        self.device = device
        self.F = F
        self.T = T

    def fold_unet_inputs(self, x):
        time_dim = x.size(-1)
        pad_len = math.ceil(time_dim / self.T) * self.T - time_dim
        padded = F.pad(x, (0, pad_len))
        if time_dim < self.T:
            return padded
        out = torch.cat(torch.split(padded, self.T, dim=-1), dim=0)
        return out

    def unfold_unet_outputs(self, x, input_size):
        batch_size, n_frames = input_size[0], input_size[-1]
        if x.size(0) == batch_size:
            return x[..., :n_frames]
        x = torch.cat(torch.split(x, batch_size, dim=0), dim=-1)
        return x[..., :n_frames]

    def trim_freq_dim(self, x):
        return x[..., :self.F, :]

    def pad_freq_dim(self, x):
        padding = (self.n_fft // 2 + 1) - x.size(-2)
        x = F.pad(x, (0, 0, 0, padding))
        return x

    def pad_stft_input(self, x):
        pad_len = (-(x.size(-1) - self.win_length) % self.hop_length) % self.win_length
        return F.pad(x, (0, pad_len))

    def _stft(self, x):
        return torch.stft(input=x,
                          n_fft=self.n_fft,
                          window=self.hann_window,
                          win_length=self.win_length,
                          hop_length=self.hop_length,
                          center=self.center,
                          return_complex=True
                          )

    def _istft(self, x, trim_length=None):
        return torch.istft(input=x,
                           n_fft=self.n_fft,
                           window=self.hann_window,
                           win_length=self.win_length,
                           hop_length=self.hop_length,
                           center=self.center,
                           length=trim_length
                           )

    def batch_stft(self, x, pad: bool = True, return_complex: bool = False):
        x_shape = x.size()
        x = x.reshape(-1, x_shape[-1])
        if pad:
            x = self.pad_stft_input(x)
        S = self._stft(x)
        S = S.reshape(x_shape[:-1] + S.shape[-2:])
        if return_complex:
            return S
        return S.abs(), S.angle()

    def batch_istft(self, magnitude, phase, trim_length=None):
        S = torch.polar(magnitude, phase)
        S_shape = S.size()
        S = S.reshape(-1, S_shape[-2], S_shape[-1])
        x = self._istft(S, trim_length)
        x = x.reshape(S_shape[:-2] + x.shape[-1:])
        return x


class UNetEncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2),
                 relu_slope=0.2):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(num_features=out_channels)
        self.relu_slope = relu_slope
        self.activ = nn.LeakyReLU(self.relu_slope)
        self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_uniform_(m.weight, nonlinearity='leaky_relu', a=self.relu_slope)
            nn.init.zeros_(m.bias)

    def forward(self, x):
        c = self.conv(x)
        y = self.activ(self.bn(c))
        return y, c


class UNetDecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2),
                 output_padding=(1, 1), dropout=0.0):
        super().__init__()
        self.conv_trans = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                             padding=padding, output_padding=output_padding)
        self.bn = nn.BatchNorm2d(num_features=out_channels)
        self.dropout = nn.Dropout(dropout)
        self.activ = nn.ReLU()
        self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.conv_trans(x)
        x = self.bn(self.activ(x))
        x = self.dropout(x)
        return x


class UNet(nn.Module):

    def __init__(self, input_size: Tuple[int, ...] = (2, 2048, 512), power: float = 1.0, device: Optional[str] = None):
        super().__init__()

        self.input_size = input_size
        self.power = power

        audio_channels, f_size, t_size = input_size

        # Audio utility object
        self.utils = UNetUtils(F=f_size, T=t_size, device=device)

        # Frontend
        self.input_norm = nn.BatchNorm2d(f_size)

        # Encoders
        self.enc1 = UNetEncoderBlock(audio_channels, 16)
        self.enc2 = UNetEncoderBlock(16, 32)
        self.enc3 = UNetEncoderBlock(32, 64)
        self.enc4 = UNetEncoderBlock(64, 128)
        self.enc5 = UNetEncoderBlock(128, 256)
        self.enc6 = UNetEncoderBlock(256, 512)

        # Decoder
        self.dec1 = UNetDecoderBlock(512, 256, dropout=0.5)
        self.dec2 = UNetDecoderBlock(512, 128, dropout=0.5)
        self.dec3 = UNetDecoderBlock(256, 64, dropout=0.5)
        self.dec4 = UNetDecoderBlock(128, 32)
        self.dec5 = UNetDecoderBlock(64, 16)
        self.dec6 = UNetDecoderBlock(32, audio_channels)

        # Mask layer
        self.mask_layer = nn.Sequential(
            nn.Conv2d(audio_channels, audio_channels, kernel_size=(4, 4), dilation=(2, 2), padding=3),
            nn.Sigmoid()
        )

        self.init_mask_layer()

        if device is not None:
            self.to(device)

    def init_mask_layer(self) -> None:
        nn.init.kaiming_uniform_(self.mask_layer[0].weight)
        nn.init.zeros_(self.mask_layer[0].bias)

    def produce_mask(self, x: Tensor) -> Tensor:
        # Frontend
        x = x.transpose(1, 2)
        x = self.input_norm(x)
        x = x.transpose(1, 2)

        # Encoder
        d, c1 = self.enc1(x)
        d, c2 = self.enc2(d)
        d, c3 = self.enc3(d)
        d, c4 = self.enc4(d)
        d, c5 = self.enc5(d)
        _, c6 = self.enc6(d)

        # Decoder
        u = self.dec1(c6)
        u = self.dec2(torch.cat([c5, u], dim=1))
        u = self.dec3(torch.cat([c4, u], dim=1))
        u = self.dec4(torch.cat([c3, u], dim=1))
        u = self.dec5(torch.cat([c2, u], dim=1))
        u = self.dec6(torch.cat([c1, u], dim=1))

        # Masking
        mask = self.mask_layer(u)

        # Apply power
        mask = mask ** self.power

        return mask

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        input_size = x.size()
        x = self.utils.fold_unet_inputs(x)
        i = self.utils.trim_freq_dim(x)
        mask = self.produce_mask(i)
        mask = self.utils.pad_freq_dim(mask)
        x_hat = self.utils.unfold_unet_outputs(x * mask, input_size)
        mask = self.utils.unfold_unet_outputs(mask, input_size)
        return x_hat, mask


class UNetWaveform(UNet):
    """
    Convenience UNet implementation that accepts a time-domain signal, computes the STFT before invoking the UNet, and
    applies the inverse STFT to go back in the time domain.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        if x.dim() == 1:
            x = x.repeat(2, 1)
        if x.dim() == 2:
            x = x.unsqueeze(0)
        mag, phase = self.utils.batch_stft(x)
        mag_hat, mask = super().forward(mag)
        x_hat = self.utils.batch_istft(mag_hat, phase, trim_length=x.size(-1))
        return x_hat, mask
