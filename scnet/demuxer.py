# separate vocals and instruments from song
import os
import time
import numpy as np
import torch
import soundfile as sf
from .SCNet import SCNet
from .utils import convert_audio
from .apply import apply_model
import argparse
import sox
from torch.optim.swa_utils import AveragedModel  # , get_ema_multi_avg_fn
from omegaconf import OmegaConf

tfmnorm = sox.Transformer()
tfmnorm.norm()


class Seperator:
    def __init__(self, model, checkpoint_path):
        # self.separator = load_model(model, checkpoint_path)
        ema_avg = (
            lambda averaged_model_parameter, model_parameter, num_averaged: 0.1
            * averaged_model_parameter
            + 0.999 * model_parameter
        )
        emas = AveragedModel(model, avg_fn=ema_avg)  # V2.0
        ckpt = torch.load(
            checkpoint_path, weights_only=True, map_location=torch.device("cpu")
        )
        emas.load_state_dict(ckpt)
        emas.sources = model.sources
        self.separator = emas
        self.separator.eval()
        if torch.cuda.device_count():
            self.device = torch.device("cuda")
        else:
            print("WARNING, using CPU")
            self.device = torch.device("cpu")
        self.separator.to(self.device)

    @property
    def instruments(self):
        return ["bass", "drums", "other", "vocals"]

    def raise_aicrowd_error(self, msg):
        raise NameError(msg)

    def separate_music_file(self, mixed_sound_array, sample_rate):
        """
        Implements the sound separation for a single sound file
        Inputs: Outputs from soundfile.read('mixture.wav')
            mixed_sound_array
            sample_rate
        Outputs:
            separated_music_arrays: Dictionary numpy array of each separated instrument
            output_sample_rates: Dictionary of sample rates separated sequence
        """
        mix = torch.from_numpy(np.asarray(mixed_sound_array.T, np.float32))

        # convert audio to GPU
        mix = mix.to(self.device)
        mix_channels = mix.shape[0]
        # mix = convert_audio(mix, sample_rate, 44100, self.separator.audio_channels)

        b = time.time()
        # Separate
        with torch.no_grad():
            estimates = apply_model(
                self.separator, mix[None], overlap=0.5, progress=False
            )[0]

        # Printing some sanity checks.
        print(time.time() - b, mono.shape[-1] / sample_rate, mix.std(), estimates.std())

        return estimates

    def load_audio(self, file_path):
        try:
            data, sample_rate = sf.read(file_path, dtype="float32")
            return data, sample_rate
        except Exception as e:
            print(f"Error loading audio file {file_path}: {e}")
            raise

    def save_sources(self, estimates, output_sample_rates, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        sources = ["instrumental", "vocal"]
        for name, src in zip(sources, estimates):
            save_path = os.path.join(save_dir, f"{name}.wav")
            sf.write(save_path, src.T, output_sample_rates)
            print(f"Saved {name} to {save_path}")

    def process_directory(self, input_dir, output_dir):
        for entry in os.listdir(input_dir):
            entry_path = os.path.join(input_dir, entry)
            if os.path.isdir(entry_path):
                mixture_path = os.path.join(entry_path, "mixture.wav")
                if os.path.isfile(mixture_path):
                    print(f"Processing {mixture_path}")
                    entry_name = os.path.basename(entry)
                else:
                    continue
            elif os.path.isfile(entry_path) and entry_path.lower().endswith(
                (".wav", ".mp3")
            ):
                print(f"Processing {entry_path}")
                mixture_path = entry_path
                entry_name = os.path.splitext(os.path.basename(entry))[0]
            else:
                continue

            mixed_sound_array, sample_rate = self.load_audio(mixture_path)
            mix = torch.from_numpy(np.asarray(mixed_sound_array.T, np.float32))
            # preprocess
            # to 22K, 1ch
            mix = convert_audio(mix, sample_rate, 22050, channels=1)
            # normalize
            mix = tfmnorm.build_array(
                input_array=mix.squeeze().numpy(), sample_rate_in=22050
            )
            mix = torch.tensor(mix)
            # 22K-> 44K, 1ch -> 2ch
            mix = convert_audio(mix, 22050, 44100, channels=2)
            mult = 3
            mix *= mult
            estimates = apply_model(
                self.separator, mix[None], overlap=0.5, progress=False
            )[0]
            estimates = estimates / mult

            save_dir = os.path.join(output_dir, entry_name)
            self.save_sources(estimates, 44100, save_dir)


def parse_args():
    parser = argparse.ArgumentParser(description="Music Source Separation using SCNet")
    parser.add_argument(
        "--input_dir", type=str, help="Input directory containing audio files"
    )
    parser.add_argument(
        "--output_dir", type=str, help="Output directory to save separated sources"
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default="./conf/config_large.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="./result/checkpoint.th",
        help="Path to model checkpoint file",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    with open(args.config_path, "r") as file:
        config = OmegaConf.load(file)

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    model = SCNet(**config.model)
    model.eval()
    seperator = Seperator(model, args.checkpoint_path)
    seperator.process_directory(args.input_dir, args.output_dir)
