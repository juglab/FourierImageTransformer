import torch
from fast_transformers.builders import TransformerDecoderBuilder, TransformerEncoderBuilder

from fit.transformers.PositionalEncoding2D import PositionalEncoding2D
from fit.utils import convert2FC, convert_to_dft
from torch.nn import functional as F


class TRecTransformer(torch.nn.Module):
    def __init__(self,
                 d_model,
                 y_coords_proj, x_coords_proj,
                 y_coords_img, x_coords_img,
                 attention_type="linear",
                 n_layers=4,
                 n_heads=4,
                 d_query=32,
                 dropout=0.1,
                 attention_dropout=0.1):
        super(TRecTransformer, self).__init__()

        self.fourier_coefficient_embedding = torch.nn.Linear(2, d_model // 2)

        self.pos_embedding_input_projections = PositionalEncoding2D(
            d_model // 2,
            y_coords_proj,
            x_coords_proj,
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

        self.pos_embedding_target = PositionalEncoding2D(d_model, y_coords_img, x_coords_img)

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

        self.predictor = torch.nn.Linear(
            n_heads * d_query,
            2
        )

        self.conv_block = torch.nn.Sequential(
            torch.nn.Conv2d(1, d_query, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(d_query),
            torch.nn.Conv2d(d_query, d_query, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(d_query),
            torch.nn.Conv2d(d_query, 1, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, x, out_pos_emb, mag_min, mag_max, dst_flatten_coords, img_shape, attenuation):
        x = self.fourier_coefficient_embedding(x)
        x = self.pos_embedding_input_projections(x)
        z = self.encoder(x, attn_mask=None)

        output = torch.repeat_interleave(out_pos_emb, z.shape[0], dim=0)
        y_hat = self.decoder(output, z)
        y_hat = self.predictor(y_hat)

        dft_hat = convert_to_dft(y_hat, mag_min=mag_min, mag_max=mag_max, dst_flatten_coords=dst_flatten_coords,
                                  img_shape=img_shape)
        dft_hat *= attenuation
        img_hat = torch.roll(torch.fft.irfftn(dft_hat, dim=[1, 2], s=2 * (img_shape,)),
                           2 * (img_shape // 2,), (1, 2)).unsqueeze(1)
        img_post = self.conv_block(img_hat)
        img_post += img_hat

        return y_hat, img_post[:,0]
