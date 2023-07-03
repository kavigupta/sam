import torch.nn as nn


from modular_splicing.models.modules.positional_encoding import PositionalEncoding

from modular_splicing.models.modules.conv_attn.convolved_attn import ConvolvedAttention


class SingleAttentionLongRangeProcessor(nn.Module):
    """
    Uses attention to compute the influence value of each motif in the sequence.

    First applies a positional encoding to the sequence, then applies a multihead attention
    layer to the sequence. The attention layer is configured to only attend to the current
    motif and the motifs in the context window. The attention layer is configured to
    return the attention weights for each motif in the sequence.

    Shape: (batch_size, seq_len, num_motifs) -> (batch_size, seq_len, num_motifs)
    """

    def __init__(self, num_motifs, cl, num_heads=1, max_len=10_000, forward_only=False):
        super().__init__()
        self.positional_encoding = PositionalEncoding(
            num_motifs, dropout=0, max_len=max_len
        )

        self.multiheadattention = ConvolvedAttention(
            embed_dim=num_motifs,
            num_heads=num_heads,
            window=cl + 1,
            interpretable=True,
            forward_only=forward_only,
        )

    def forward(self, output, collect_intermediates):
        output = output.transpose(0, 1)
        output = self.positional_encoding(output)
        attn_output = self.multiheadattention(
            output, output, output, collect_intermediates=collect_intermediates
        )
        output = attn_output.pop("output")
        output = output.transpose(0, 1)
        return attn_output, output
