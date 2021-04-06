import torch
from fast_transformers.builders import TransformerDecoderBuilder, TransformerEncoderBuilder

from fit.transformers.PositionalEncoding2D import PositionalEncoding2D
from fit.utils import convert2DFT
from torch.nn import functional as F


class TRecTransformer(torch.nn.Module):
    def __init__(self,
                 d_model,
                 coords_sinogram, flatten_order_sinogram,
                 coords_target, flatten_order_target,
                 attention_type="linear",
                 n_layers=4,
                 n_heads=4,
                 d_query=32,
                 dropout=0.1,
                 attention_dropout=0.1,
                 d_conv=8):
        super(TRecTransformer, self).__init__()

        self.fourier_coefficient_embedding = torch.nn.Linear(2, d_model // 2)

        self.pos_embedding_input_projections = PositionalEncoding2D(
            d_model // 2,
            coords=coords_sinogram,
            flatten_order=flatten_order_sinogram,
            persistent=False
        )

        self.encoder = TransformerEncoderBuilder.from_kwargs(
            attention_type=attention_type,
            n_layers=n_layers,
            n_heads=n_heads,
            feed_forward_dimensions=n_heads * d_query * 4,
            query_dimensions=d_query,
            value_dimensions=d_query,
            dropout=dropout,
            attention_dropout=attention_dropout
        ).get()

        self.fbp_fourier_coefficient_embedding = torch.nn.Linear(2, d_model // 2)

        self.pos_embedding_target = PositionalEncoding2D(d_model // 2, coords=coords_target,
                                                         flatten_order=flatten_order_target)

        self.decoder = TransformerDecoderBuilder.from_kwargs(
            self_attention_type=attention_type,
            cross_attention_type=attention_type,
            n_layers=n_layers,
            n_heads=n_heads,
            feed_forward_dimensions=n_heads * d_query * 4,
            query_dimensions=d_query,
            value_dimensions=d_query,
            dropout=dropout,
            attention_dropout=attention_dropout
        ).get()

        self.predictor_amp = torch.nn.Linear(
            n_heads * d_query,
            1
        )
        self.predictor_phase = torch.nn.Linear(
            n_heads * d_query,
            1
        )

        self.conv_block = torch.nn.Sequential(
            torch.nn.Conv2d(1, d_conv, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(d_conv),
            torch.nn.Conv2d(d_conv, 1, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, x, fbp, amp_min, amp_max, dst_flatten_coords, img_shape, attenuation):
        x = self.fourier_coefficient_embedding(x)
        x = self.pos_embedding_input_projections(x)
        z = self.encoder(x, attn_mask=None)

        fbp = self.fbp_fourier_coefficient_embedding(fbp)
        fbp = self.pos_embedding_target(fbp)
        y_hat = self.decoder(fbp, z)
        y_amp = self.predictor_amp(y_hat)
        y_phase = F.tanh(self.predictor_phase(y_hat))
        y_hat = torch.cat([y_amp, y_phase], dim=-1)

        dft_hat = convert2DFT(y_hat, amp_min=amp_min, amp_max=amp_max, dst_flatten_order=dst_flatten_coords,
                              img_shape=img_shape)
        dft_hat *= attenuation
        img_hat = torch.roll(torch.fft.irfftn(dft_hat, dim=[1, 2], s=2 * (img_shape,)),
                             2 * (img_shape // 2,), (1, 2)).unsqueeze(1)
        img_post = self.conv_block(img_hat)
        img_post += img_hat

        return y_hat, img_post[:, 0]


class TRecOnlyFBP(torch.nn.Module):
    def __init__(self,
                 d_model,
                 coords_target, flatten_order_target,
                 attention_type="linear",
                 n_layers=4,
                 n_heads=4,
                 d_query=32,
                 dropout=0.1,
                 attention_dropout=0.1,
                 d_conv=8):
        super(TRecOnlyFBP, self).__init__()

        self.fbp_fourier_coefficient_embedding = torch.nn.Linear(2, d_model // 2)

        self.pos_embedding_target = PositionalEncoding2D(d_model // 2, coords=coords_target,
                                                         flatten_order=flatten_order_target)

        self.encoder = TransformerEncoderBuilder.from_kwargs(
            attention_type=attention_type,
            n_layers=n_layers,
            n_heads=n_heads,
            feed_forward_dimensions=n_heads * d_query * 4,
            query_dimensions=d_query,
            value_dimensions=d_query,
            dropout=dropout,
            attention_dropout=attention_dropout
        ).get()

        self.predictor_amp = torch.nn.Linear(
            n_heads * d_query,
            1
        )
        self.predictor_phase = torch.nn.Linear(
            n_heads * d_query,
            1
        )

        self.conv_block = torch.nn.Sequential(
            torch.nn.Conv2d(1, d_conv, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(d_conv),
            torch.nn.Conv2d(d_conv, 1, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, x, fbp, mag_min, mag_max, dst_flatten_coords, img_shape, attenuation):
        fbp = self.fbp_fourier_coefficient_embedding(fbp)
        fbp = self.pos_embedding_target(fbp)
        y_hat = self.encoder(fbp)
        y_amp = self.predictor_amp(y_hat)
        y_phase = F.tanh(self.predictor_phase(y_hat))
        y_hat = torch.cat([y_amp, y_phase], dim=-1)

        dft_hat = convert2DFT(y_hat, amp_min=mag_min, amp_max=mag_max, dst_flatten_order=dst_flatten_coords,
                              img_shape=img_shape)
        dft_hat *= attenuation
        img_hat = torch.roll(torch.fft.irfftn(dft_hat, dim=[1, 2], s=2 * (img_shape,)),
                             2 * (img_shape // 2,), (1, 2)).unsqueeze(1)
        img_post = self.conv_block(img_hat)
        img_post += img_hat

        return y_hat, img_post[:, 0]


class TRecOnlyConvBlock(torch.nn.Module):
    def __init__(self, d_conv=8):
        super(TRecOnlyConvBlock, self).__init__()

        self.conv_block = torch.nn.Sequential(
            torch.nn.Conv2d(1, d_conv, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(d_conv),
            torch.nn.Conv2d(d_conv, 1, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, x, fbp, mag_min, mag_max, dst_flatten_coords, img_shape, attenuation):
        dft_hat = convert2DFT(fbp, amp_min=mag_min, amp_max=mag_max, dst_flatten_order=dst_flatten_coords,
                              img_shape=img_shape)
        dft_hat *= attenuation
        img_hat = torch.roll(torch.fft.irfftn(dft_hat, dim=[1, 2], s=2 * (img_shape,)),
                             2 * (img_shape // 2,), (1, 2)).unsqueeze(1)
        img_post = self.conv_block(img_hat)
        img_post += img_hat

        return fbp, img_post[:, 0]
