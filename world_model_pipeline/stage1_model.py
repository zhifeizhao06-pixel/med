import torch
import torch.nn as nn


class Stage1SegResNetVAE(nn.Module):
    """Thin, stable adapter around MONAI SegResNetVAE.

    MONAI returns (segmentation_logits, vae_loss) while training and only
    segmentation logits in eval mode. The adapter normalizes that contract.
    """

    def __init__(self, roi_size=(96, 96, 96), in_channels=4, out_channels=1, init_filters=16, vae_nz_dim=256):
        super().__init__()
        try:
            from monai.networks.nets import SegResNetVAE
        except ImportError as exc:
            raise ImportError("Install MONAI before constructing Stage1SegResNetVAE") from exc
        self.network = SegResNetVAE(
            input_image_size=tuple(roi_size),
            in_channels=in_channels,
            out_channels=out_channels,
            init_filters=init_filters,
            blocks_down=(1, 2, 2, 4),
            blocks_up=(1, 1, 1),
            vae_nz_dim=vae_nz_dim,
            vae_estimate_std=True,
        )

    def forward(self, image):
        output = self.network(image)
        if isinstance(output, tuple):
            logits, vae_loss = output
        else:
            logits = output
            vae_loss = logits.new_zeros(())
        return logits, vae_loss

