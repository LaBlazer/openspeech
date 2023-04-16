import torch
import torch.nn as nn
from torch import Tensor
import torchaudio
import numpy as np
import librosa
import hydra
import os

from openspeech.models import MODEL_REGISTRY
from openspeech.tokenizers import TOKENIZER_REGISTRY
from openspeech.dataclass.initialize import hydra_eval_init
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.utilities import rank_zero_info

def parser(signal, audio_extension: str = 'wav') -> Tensor:
    feature = torchaudio.compliance.kaldi.fbank(
        waveform=Tensor(signal[0]).unsqueeze(0),
        num_mel_bins=80,
        frame_length=20,
        frame_shift=10,
        window_type='hamming'
    ).transpose(0, 1).numpy()

    feature -= feature.mean()
    feature /= np.std(feature)

    return torch.FloatTensor(feature).transpose(0, 1)


@hydra.main(config_path=os.path.join("."), config_name="eval.yml")
def hydra_main(configs: DictConfig) -> None:
    rank_zero_info(OmegaConf.to_yaml(configs))
    use_cuda = configs.eval.use_cuda and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    tokenizer = TOKENIZER_REGISTRY[configs.tokenizer.unit](configs)

    model = MODEL_REGISTRY[configs.model.model_name]
    model = model.load_from_checkpoint(configs.eval.checkpoint_path, configs=configs, tokenizer=tokenizer)
    model.to(device)
    signal = librosa.load('test.wav', sr=16000)
    feature = parser(signal)

    with torch.no_grad():
        outputs = model(feature.unsqueeze(0).to('cuda'), torch.Tensor([feature.shape[0]]).to('cuda'))
    print(outputs)
    prediction = tokenizer.decode(outputs["predictions"].cpu().detach().numpy())
    print(prediction)

if __name__ == '__main__':
    hydra_eval_init()
    hydra_main()