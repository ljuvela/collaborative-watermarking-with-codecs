import torch

from ..third_party.asvspoof2021.rawnet2.model import RawNet as RawNetBase

from torchaudio.transforms import Resample

class RawNet2(RawNetBase):

    def __init__(
            self,
            sample_rate=16000,
            first_conv=1024,
            in_channels=1,
            filts=[20, [20, 20], [20, 128], [128, 128]],
            nb_fc_node= 1024,
            gru_node=1024,
            nb_gru_layer=3,
            nb_classes=2,
            device=torch.device('cpu'),
            use_batch_norm=True,
            pad_input_to_len: int = None
            ):
        """
        Args:
            first_conv: no. of filter coefficients 
            in_channels: ?
            filts: no. of filters channel in residual blocks
            nb_fc_node: ?
            gru_node: ?
            nb_gru_layer: ?
            nb_classes: ?
            pad_input_to_len: pad input to specific length (default: None, uses input as is)
        
        """
        d_args = {
            'first_conv': first_conv,
            'in_channels': in_channels,
            'filts': filts,
            'nb_fc_node': nb_fc_node,
            'gru_node': gru_node,
            'nb_gru_layer': nb_gru_layer,
            'nb_classes': nb_classes
        }


        super().__init__(d_args=d_args, device=device, 
                         use_batch_norm=use_batch_norm)

        self.sample_rate = sample_rate
        if self.sample_rate != 16000:
            self.resampler = Resample(orig_freq=self.sample_rate, new_freq=16000)
        else:
            self.resampler = None

        self.pad_input_to_len = pad_input_to_len


    def eval(self, pass_gradients=True):
        """
        Set model to eval mode

        cuDNN RNNs are not passing gradients in eval mode 
        and these need to be exempted if gradient flow is needed
        """
        if not pass_gradients:
            self.eval()
        else:
            for name, module in self.named_children():
                if name != 'gru':
                    module.eval()
            self.training = False


    def forward(self, x):
        """
        Args:
            x: (batch, channels=1, length)

        Returns:
            scores: (batch, length=1)

        """

        if x.ndim != 3: 
            raise ValueError(f"Expected input of shape (batch, channels=1, timestesps), got {x.shape}")
        if x.size(1) != 1:
            raise ValueError(f"Expected single channel input, got {x.shape}")

        if self.resampler is not None:
            x = self.resampler(x)

        if self.pad_input_to_len is not None:
            # left padding
            x = torch.nn.functional.pad(x, pad=(self.pad_input_to_len - x.size(-1), 0), mode='constant', value=0.0)

        log_out = super().forward(x[:, 0, :])

        # slice from (batch, num_classes) -> (batch, 1)
        log_out = log_out[:, 0:1]

        return torch.exp(log_out)