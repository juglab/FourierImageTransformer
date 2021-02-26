import torch
from fast_transformers.builders import TransformerEncoderBuilder
from fast_transformers.masking import TriangularCausalMask

from fit.transformers.PositionalEncoding2D import PositionalEncoding2D


class SResTransformer(torch.nn.Module):
    def __init__(self,
                 d_model,
                 y_coords_img, x_coords_img, flatten_order,
                 attention_type="linear",
                 n_layers=4,
                 n_heads=4,
                 d_query=32,
                 dropout=0.1,
                 attention_dropout=0.1):
        super(SResTransformer, self).__init__()

        self.fourier_coefficient_embedding = torch.nn.Linear(2, d_model // 2)

        self.pos_embedding_input_projections = PositionalEncoding2D(
            d_model // 2,
            y_coords_img,
            x_coords_img,
            flatten_order=flatten_order,
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

        self.predictor = torch.nn.Linear(
            n_heads * d_query,
            2
        )

    def forward(self, x):
        x = self.fourier_coefficient_embedding(x)
        x = self.pos_embedding_input_projections(x)
        triangular_mask = TriangularCausalMask(x.shape[1], device=x.device)
        y_hat = self.encoder(x, attn_mask=triangular_mask)
        y_hat = self.predictor(y_hat)

        return y_hat
