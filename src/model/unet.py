from torch import nn
#TODO: add implementation of group UNet to leverage equivariance
#https://github.com/dogeplusplus/group-unet/blob/main/group_unet/layers.py#L93

class UNet(nn.Module):
    """
    A minimal implementation of the Unet architecture.

    Encoder: comprises 3 convolutional layers,
    each succeeded with a ReLU activation.

    Decoder: comprises 3 convolutional layers,
    first two succeeded with a ReLU activation.

    self.time_mlp: embeds the time signal, linearly projecting it
    into a 256-dimensional representation.

    self.batch_normalization: normalizes input batches.
    """
    def __init__(self, in_channels=3, out_channels=3, time_emb_dim=256):
        super().__init__()

        # time embedding block
        self.time_mlp = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, time_emb_dim),
            nn.ReLU()
        )

        # simple encoder block
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU()
        )

        # simple decoder block
        self.decoder = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, out_channels, kernel_size=3, padding=1)
        )
        self.batch_norm = nn.BatchNorm2d(time_emb_dim)


    def forward(self, x, t):
        """
        performs the forward pass of the model,
        the input is first permuted to channels first shape,
        the input is passed through the encoder layers,
        at the bottleneck, the time embedding is added,
        then the decoder layers are applied, and the output is reshaped
        back to the original form.
        """
        # permute to channels first
        x = x.permute(0, 3, 1, 2)
        t_emb = self.time_mlp(t.float()).unsqueeze(-1).unsqueeze(-1)

        x = self.batch_norm(self.encoder(x) + t_emb)
        x = self.decoder(x)

        # permute back to original shape (256x37x3)
        x = x.permute(0,2,3,1)
        return x