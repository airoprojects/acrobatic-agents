

class VAE(torch.nn.Module):
    def __init__(self, length, nclasses, latent_size, transition_channels):
        super(VAE, self).__init__()
        self.encoder = Encoder(1, length, nclasses, latent_size, transition_channels)
        self.decoder = Decoder(length, transition_channels, nclasses, latent_size)
    def count_parameters(self):
        return np.sum([np.prod(x.size()) for x in self.parameters()])
    def forward(self, x):
        oh_class, mean, z = self.encoder(x)
        x_decoded = self.decoder(z)
        return oh_class, mean, z, x_decoded