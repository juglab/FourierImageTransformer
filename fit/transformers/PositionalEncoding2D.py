import math
import torch


class PositionalEncoding2D(torch.nn.Module):
    """
    Positional encoding based on Fourier features first introduced by Vaswani et al.[1] and
    extended to 2D by Wang et al.[2].

    We adapted it to work with arbitrary real coordinates. The coordinates can be from any 2D coordinate system e.g.
    cartesian or polar.

    References:
        [1] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and
        Illia Polosukhin.
        Attention is all you need.
        In Advances in neural information processing systems, pages 5998–6008, 2017.
        [2]  Zelun Wang and Jyh-Charn Liu.
        Translating math formula images to latex sequences using deep neural networks with sequence-level training.
        International Journal on Document Analysis and Recognition (IJDAR), pages 1–13, 2020.
    """

    def __init__(self, dimensionality, coords, flatten_order, dropout=0.0, persistent=False):
        """
        Create a Fourier feature based 2D positional encoding. The positional encoding will be flattened
        according to the `flatten_order`, hence `forward` assumes that the input `x` is flattened accordingly.

        :param dimensionality: of the encoding
        :param coords: 2D coordinates (y, x) for cartesian or (r, phi) for polar
        :param flatten_order: order use for input flattening
        :param dropout: applied to the positional encoding
        :param persistent: of the positional encoding
        """
        super(PositionalEncoding2D, self).__init__()
        self.dropout = torch.nn.Dropout(p=dropout)
        self.d_model = dimensionality

        pe = self._positional_encoding_2D(self.d_model, coords)
        pe = torch.movedim(pe, 0, -1).unsqueeze(0)
        pe = pe[:, flatten_order]

        self.register_buffer('pe', pe, persistent=persistent)

    def _positional_encoding_2D(self, d_model, coords):
        pe = torch.zeros(d_model, coords[0].shape[0])

        d_model = int(d_model / 2)
        div_term = torch.exp(torch.arange(0., d_model, 2) *
                             -(math.log(10000.0) / d_model))

        pe[0:d_model:2, :] = torch.sin(coords[0].unsqueeze(-1) * div_term).permute(-1, 0)
        pe[1:d_model:2, :] = torch.cos(coords[0].unsqueeze(-1) * div_term).permute(-1, 0)
        pe[d_model::2, :] = torch.sin(coords[1].unsqueeze(-1) * div_term).permute(-1, 0)
        pe[d_model + 1::2, :] = torch.cos(coords[1].unsqueeze(-1) * div_term).permute(-1, 0)

        return pe

    def forward(self, x):
        pos_embedding = self.pe[:, :x.size(1), :]
        pos_embedding = torch.repeat_interleave(pos_embedding, x.shape[0], dim=0)
        x = torch.cat([x, pos_embedding], dim=2)
        return self.dropout(x)

    def forward_i(self, x, i):
        pos_embedding = self.pe[0, i:i + 1]
        x = torch.cat([x, pos_embedding.expand_as(x)], dim=1)
        return self.dropout(x)
