import torch


from .lfcc_lcnn import LFCC_LCNN
from .rawnet2 import RawNet2

class WatermarkModel(torch.nn.Module):

    def __init__(self, model_type:str, sample_rate:int, config):
        super().__init__()

     
        self.model_type = model_type

        if model_type == "lfcc_lcnn":
            self.model = LFCC_LCNN(
                in_dim=1, out_dim=1, 
                sample_rate=sample_rate, 
                sigmoid_output=config.get('lfcc_lcnn_sigmoid_out', True),
                dropout_prob=config.get('lfcc_lcnn_dropout_prob', 0.7),
                use_batch_norm=config.get('lfcc_lcnn_use_batch_norm', True)
                )
        elif model_type == 'raw_net':
            self.model = RawNet2(
                sample_rate=sample_rate,
                use_batch_norm=config.get('raw_net_use_batch_norm', True),
                pad_input_to_len=config.get('raw_net_input_pad_len', None)
            )
        else:
            raise ValueError(f"Unsupported watermark model type {model_type}")


    def forward(self, x_real, x_fake):

        y_real = self.model(x_real)
        y_fake = self.model(x_fake)

        return y_real, y_fake


    def load_pretrained_state_dict(self, state_dict):

        if self.model_type == 'lfcc_lcnn':
            state_dict_old = self.models['lfcc_lcnn'].state_dict()
            optional_keys = ['resampler.kernel']
            for ok in optional_keys:
                val = state_dict.get(ok, state_dict_old[ok])
                state_dict[ok] = val
            self.models['lfcc_lcnn'].load_state_dict(state_dict)
        elif self.model_type == 'raw_net':
            state_dict_old = self.models['raw_net'].state_dict()
            optional_keys = ['resampler.kernel']
            for ok in optional_keys:
                val = state_dict.get(ok, state_dict_old[ok])
                state_dict[ok] = val
            self.models['raw_net'].load_state_dict(state_dict)
        else:
            raise NotImplementedError()

    def output_layer_requires_grad_(self, requires_grad: bool = True):

        if self.model_type == 'lfcc_lcnn':
            self.models['lfcc_lcnn'].m_output_act.requires_grad_(requires_grad)
        elif self.model_type == 'raw_net':
            self.models['raw_net'].fc2_gru.requires_grad_(requires_grad)
        else:
            raise NotImplementedError()



