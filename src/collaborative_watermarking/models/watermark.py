import torch
import typing
from pathlib import Path

from audiotools.ml import BaseModel

from .lfcc_lcnn import LFCC_LCNN
from .rawnet2 import RawNet2

class WatermarkModel(BaseModel):

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


    def save_to_folder(
        self,
        folder: typing.Union[str, Path],
        extra_data: dict = None,
        package: bool = True,
    ):
        """Dumps a model into a folder, as both a package
        and as weights, as well as anything specified in
        ``extra_data``. ``extra_data`` is a dictionary of other
        pickleable files, with the keys being the paths
        to save them in. The model is saved under a subfolder
        specified by the name of the class (e.g. ``folder/generator/[package, weights].pth``
        if the model name was ``Generator``).

        >>> with tempfile.TemporaryDirectory() as d:
        >>>     extra_data = {
        >>>         "optimizer.pth": optimizer.state_dict()
        >>>     }
        >>>     model.save_to_folder(d, extra_data)
        >>>     Model.load_from_folder(d)

        Parameters
        ----------
        folder : typing.Union[str, Path]
            _description_
        extra_data : dict, optional
            _description_, by default None

        Returns
        -------
        str
            Path to folder
        """
        extra_data = {} if extra_data is None else extra_data
        model_name = type(self).__name__.lower()
        target_base = Path(f"{folder}/{model_name}/")
        target_base.mkdir(exist_ok=True, parents=True)

        if package:
            package_path = target_base / f"package.pth"
            self.save(package_path)

        weights_path = target_base / f"weights.pth"
        self.save(weights_path, package=False)

        for path, obj in extra_data.items():
            torch.save(obj, target_base / path)

        return target_base

    @classmethod
    def load_from_folder(
        cls,
        folder: typing.Union[str, Path],
        package: bool = True,
        strict: bool = False,
        **kwargs,
    ):
        """Loads the model from a folder generated by
        :py:func:`audiotools.ml.layers.base.BaseModel.save_to_folder`.
        Like that function, this one looks for a subfolder that has
        the name of the class (e.g. ``folder/generator/[package, weights].pth`` if the
        model name was ``Generator``).

        Parameters
        ----------
        folder : typing.Union[str, Path]
            _description_
        package : bool, optional
            Whether to use ``torch.package`` to load the model,
            loading the model from ``package.pth``.
        strict : bool, optional
            Ignore unmatched keys, by default False

        Returns
        -------
        tuple
            tuple of model and extra data as saved by
            :py:func:`audiotools.ml.layers.base.BaseModel.save_to_folder`.
        """
        folder = Path(folder) / cls.__name__.lower()
        model_pth = "package.pth" if package else "weights.pth"
        model_pth = folder / model_pth

        model = cls.load(model_pth, strict=strict)
        extra_data = {}
        excluded = ["package.pth", "weights.pth"]
        files = [x for x in folder.glob("*") if x.is_file() and x.name not in excluded]
        for f in files:
            extra_data[f.name] = torch.load(f, **kwargs)

        return model, extra_data
