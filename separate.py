from larsnet import LarsNet
from pathlib import Path
from typing import Union, Optional
import soundfile as sf
import argparse


def separate(input_dir: Union[str, Path], output_dir: Union[str, Path], wiener_exponent: Optional[float], device: str):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    if not input_dir.exists():
        raise RuntimeError(f'{input_dir} was not found.')

    if wiener_exponent is not None and wiener_exponent <= 0:
        raise ValueError(f'α-Wiener filter exponent should be positive.')

    larsnet = LarsNet(
        wiener_filter=wiener_exponent is not None,
        wiener_exponent=wiener_exponent,
        device=device,
        config="config.yaml",
    )

    for mixture in input_dir.rglob("*.wav"):

        stems = larsnet(mixture)

        for stem, waveform in stems.items():
            save_path = output_dir.joinpath(stem, f'{mixture.stem}.wav')
            save_path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(save_path, waveform.cpu().numpy().T, larsnet.sr)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_dir', type=str, required=True, help="Path to the root directory where to find the target drum mixtures.")
    parser.add_argument('-o', '--output_dir', type=str, default='separated_stems', help="Path to the directory where to save the separated tracks.")
    parser.add_argument('-w', '--wiener_exponent', type=float, default=None, help="Positive α-Wiener filter exponent (float). Use it only if Wiener filtering is to be applied.")
    parser.add_argument('-d', '--device', type=str, default='cpu', help="Torch device. Default 'cpu'")

    args = vars(parser.parse_args())

    separate(
        input_dir=args['input_dir'],
        output_dir=args['output_dir'],
        wiener_exponent=args['wiener_exponent'],
        device=args['device']
    )
