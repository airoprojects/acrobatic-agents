import torch
import torch.nn as nn

class BCOFormer(nn.Module):
    def __init__(self, input_dim, output_dim, nhead, num_encoder_layers, num_decoder_layers):
        super(BCOFormer, self).__init__()
        self.transformer = nn.Transformer(d_model=input_dim, nhead=nhead, 
                                          num_encoder_layers=num_encoder_layers, 
                                          num_decoder_layers=num_decoder_layers)
        self.out = nn.Linear(input_dim, output_dim)

    def forward(self, src, tgt):
        transformer_output = self.transformer(src, tgt)
        return self.out(transformer_output)

# Example usage
model = BCOFormer(input_dim=197, output_dim=36, nhead=5, num_encoder_layers=2, num_decoder_layers=2)
src = torch.rand(10, 32, 197)  # (sequence length, batch size, feature size)
tgt = torch.rand(20, 32, 197)  # target sequence
output = model(src, tgt)
