# Deep Drums Demixing 🥁

We introduce **StemGMD**, a large-scale multi-kit audio dataset of isolated single-instrument drum stems. Each audio clip is synthesized from MIDI recordings of expressive drums performances from Magenta's [Groove MIDI Dataset](https://magenta.tensorflow.org/datasets/groove) using ten real-sounding acoustic drum kits. 

Totaling **1224 hours of audio**, StemGMD is the largest dataset of drums to date and the first to comprise isolated audio clips for every instrument in a canonical nine-piece drum kit.

We leverage StemGMD to develop and release **LarsNet**, a new deep drums demixing model that can separate five stems from a stereo drum mixture faster than real-time using a parallel arrangement of dedicated U-Nets.

📜 The paper "_Toward Deep Drum Source Separation_" authored by A. I. Mezza, R. Giampiccolo, A. Bernardini, and A. Sarti has been submitted to *Pattern Recognition Letters*.

## StemGMD 🎵
**StemGMD is freely available on [Zenodo](https://zenodo.org/records/7860223) under the CC-BY 4.0 license.**

StemGMD was created by taking all the MIDI recordings in Groove MIDI Dataset, applying a MIDI mapping reducing the number of channels from 22 down to 9, and then manually synthetizing the isolated tracks as 16bit/44.1kHz WAV files with 10 different acoustic drum kits using Apple's Drum Kit Designer in Logic Pro X.

StemGMD contains isolated stems of 9 canonical drum pieces:
- Kick Drum
- Snare
- High Tom
- Low-Mid Tom
- High Floor Tom
- Closed Hi-Hat
- Open Hi-Hat
- Crash Cymbal
- Ride Cymbal

These stems were obtained applying the MIDI mapping described in Appendix B of [(Gillick et al., 2019)](https://arxiv.org/abs/1905.06118).

## Separate a drum track using LarsNet

First, download the pretrained models ⚠️ **[soon available]** ⚠️

Then, modify the `inference_models` paths in `config.py` as needed.

Finally, on your terminal, simply run 

`$ python separate -i /path/to/the/folder/containing/your/audio/files` 

By default, the script will create a folder named `separated_stems` where to save the results. Alternatively, you can specify the output directory as follows:

`$ python separate -i /path/to/the/folder/containing/your/audio/files -o /path/to/output/folder/` 

Optionally, you can run a LarsNet version implementing α-Wiener filtering. This is done by specifying the option `-w` followed by a postive floating-point number indicating the exponent α to be applied: 

`$ python separate -i /path/to/the/folder/containing/your/audio/files -w 1.0` 

## Audio Examples 🎧
Audio examples are available on our [GitHub page](https://polimi-ispl.github.io/larsnet/)

## Beyond Drums Demixing
The structure of StemGMD follows that of Magenta's Groove MIDI Dataset (GMD). Therefore, GMD metadata is preserved in StemGMD, including `drummer`, `session`, `style`, `bpm`, `beat_type`, `time_signature`, `duration`, `split`. 

This extends the applications of StemGMD beyond Deep Drums Demixing.

In fact, we argue that StemGMD can rival other large-scale datasets, such as **Expanded Groove MIDI Dataset** ([E-GMD](https://arxiv.org/abs/2004.00188)), for tasks such as Automatic Drum Transcription when considering the countless possbilities for data augmentation that having isolated stems allows for.
