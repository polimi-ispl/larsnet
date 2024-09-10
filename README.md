# Deep Drum Source Separation 🥁

We introduce **StemGMD**, a large-scale multi-kit audio dataset of isolated single-instrument drum stems. Each audio clip is synthesized from MIDI recordings of expressive drums performances from Magenta's [Groove MIDI Dataset](https://magenta.tensorflow.org/datasets/groove) using ten real-sounding acoustic drum kits. 

Totaling **1224 hours of audio**, StemGMD is the largest dataset of drums to date and the first to comprise isolated audio clips for every instrument in a canonical nine-piece drum kit.

We leverage StemGMD to develop and release **LarsNet**, a new deep drums demixing model that can separate five stems from a stereo drum mixture faster than real-time using a parallel arrangement of dedicated U-Nets.

📝 The open access paper "_**Toward Deep Drum Source Separation**_" authored by A. I. Mezza, R. Giampiccolo, A. Bernardini, and A. Sarti has been published on *Pattern Recognition Letters*: [https://doi.org/10.1016/j.patrec.2024.04.026](https://doi.org/10.1016/j.patrec.2024.04.026)

```
@article{larsnet,
  title = {Toward deep drum source separation},
  journal = {Pattern Recognition Letters},
  volume = {183},
  pages = {86-91},
  year = {2024},
  issn = {0167-8655},
  doi = {https://doi.org/10.1016/j.patrec.2024.04.026},
  url = {https://www.sciencedirect.com/science/article/pii/S0167865524001351},
  author = {Alessandro Ilic Mezza and Riccardo Giampiccolo and Alberto Bernardini and Augusto Sarti}
}
```

## StemGMD 🎵
**StemGMD is freely available on [Zenodo](https://zenodo.org/records/7860223) under the CC-BY 4.0 license.**

StemGMD was created by taking all the MIDI recordings in Groove MIDI Dataset, applying a MIDI mapping reducing the number of channels from 22 down to 9, and then manually synthetizing the isolated tracks as 16bit/44.1kHz WAV files with ten different acoustic drum kits using Apple's Drum Kit Designer in Logic Pro X.

StemGMD contains isolated stems of nine canonical drum pieces:
- **Kick Drum**
- **Snare**
- **High Tom**
- **Low-Mid Tom**
- **High Floor Tom**
- **Closed Hi-Hat**
- **Open Hi-Hat**
- **Crash Cymbal**
- **Ride Cymbal**

These stems were obtained by applying the MIDI mapping described in Appendix B of [(Gillick et al., 2019)](https://arxiv.org/abs/1905.06118).

## LarsNet 🥁

LarsNet can separate five stems from a stereo drum mixture:
- **Kick Drum**
- **Snare**
- **Tom-Toms** (High, Mid-Low, and Floor tom)
- **Hi-Hat** (Open and Closed Hi-Hat)
- **Cymbals** (Crash and Ride Cymbals)

## Pretrained LarsNet Models 📥 

Pretrained LarsNet model checkpoints can be found [here](https://drive.google.com/uc?id=1U8-5924B1ii1cjv9p0MTPzayb00P4qoL&export=download) (562 MB) licensed under CC BY-NC 4.0.

## Separate a drum track using LarsNet ✂️

First, [download](https://drive.google.com/uc?id=1U8-5924B1ii1cjv9p0MTPzayb00P4qoL&export=download) the pretrained models.   

Then, unzip the folder and place it in the project directory. Alternatively, modify the `inference_models` paths in `config.py` as needed.

Finally, run the following command on your terminal:

`$ python separate.py -i /path/to/the/folder/containing/your/audio/files` 

By default, the script will create a folder named `separated_stems` where to save the results. Alternatively, you can specify the output directory by using the `-o` option:

`$ python separate.py -i /path/to/the/folder/containing/your/audio/files -o /path/to/output/folder/` 

Optionally, you can run a LarsNet version implementing [α-Wiener filtering](https://doi.org/10.1109/ICASSP.2015.7177973) by specifying the option `-w` followed by a postive floating-point number indicating the exponent α to be applied, e.g., 

`$ python separate.py -i /path/to/the/folder/containing/your/audio/files -w 1.0` 

This latter version is expected to reduce cross-talk artifacts between separated stems, but might introduce side-chain compression-like artifacts. Namely, choosing α∊(0, 1) would result in more bleed, whereas α≥1 risk increasing the so-called ducking effect.

Lastly, you can specify the device using the `-d` option (default: `cpu`)

`$ python separate.py -i /path/to/the/folder/containing/your/audio/files -d cuda:0` 

## Train LarsNet 🛠️

We are working toward releasing the scripts for fine-tuning and training LarsNet from scratch. The code will be available soon.

## Audio Examples 🎧
Audio examples are available on our [GitHub page](https://polimi-ispl.github.io/larsnet/)

## Beyond Drums Demixing 🔜
The structure of StemGMD follows that of Magenta's Groove MIDI Dataset (GMD). Therefore, GMD metadata is preserved in StemGMD, including annotations such as `drummer`, `session`, `style`, `bpm`, `beat_type`, `time_signature`, `split`, as well as the source MIDI data. 

This extends the applications of StemGMD beyond Deep Drums Demixing.

In fact, we argue that StemGMD may rival other large-scale datasets, such as **Expanded Groove MIDI Dataset** ([E-GMD](https://arxiv.org/abs/2004.00188)), for tasks such as Automatic Drum Transcription when considering the countless possbilities for data augmentation that having isolated stems allows for.

## LARS 🔌

You may also want to check out **LARS**, an open-source VST3/AU plug-in that runs LarsNet under the hood and can be used inside any DAW.

LARS was presented at ISMIR 2023 Late-Breaking Demo Session
> A. I. Mezza, R. di Palma, E. Morena, A. Orsatti, R. Giampiccolo, A. Bernardini, and A. Sarti, "LARS: An open-source VST3 plug-in for deep drums demixing with pretrained models," _ISMIR 2023 LBD Session_, 2023.

:pencil: [LP-33: LARS: An open-source VST3 plug-in for deep drums demixing with pretrained models](https://ismir2023program.ismir.net/lbd_349.html)

:link: [LARS GitHub repository](https://github.com/EdoardoMor/LARS)
