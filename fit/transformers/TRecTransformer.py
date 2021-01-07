import torch
from fast_transformers.builders import TransformerDecoderBuilder, TransformerEncoderBuilder

from fit.transformers.PositionalEncoding2D import PositionalEncoding2D


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

    def forward(self, x, out_pos_emb):
        x = self.fourier_coefficient_embedding(x)
        x = self.pos_embedding_input_projections(x)
        z = self.encoder(x, attn_mask=None)

        output = torch.repeat_interleave(out_pos_emb, z.shape[0], dim=0)
        y_hat = self.decoder(output, z)
        y_hat = self.predictor(y_hat)

        return y_hat
