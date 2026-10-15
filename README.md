# Music Genre Classification

A GTZAN music genre classifier comparing SVM, distance-weighted k-NN, a feed-forward neural network, and CNNs using raw waveform and log-mel inputs. A local Streamlit app uses a validation-selected mel CNN and averages five windows per recording.

## Quick start on this laptop

```bash
cd /home/bae/sem5/ml/miniproject/music_recommendation
scripts/run.sh gpu-check
scripts/run.sh app
```

Open the local address printed by Streamlit. Upload a WAV or FLAC file with at least two seconds of music. The app processes up to the first 30 seconds. Neural training and app inference require the NVIDIA GPU.

## Measured results

The selected mel CNN achieves 64% accuracy on central two-second test windows. Averaging five windows per recording achieves 70% accuracy and 0.706 macro-F1 on the 100 held-out test recordings. These are separate evaluation protocols. See `docs/RESULTS.md` and the auditable prediction records for all ten experiments.

## Reproduce in a fresh environment

Use Linux, Python 3.12, a working NVIDIA driver, and enough storage for the dataset, TensorFlow, CUDA dependencies, caches, and trained models.

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-gpu.txt
scripts/run.sh gpu-check
scripts/run.sh download
scripts/run.sh prepare
scripts/run.sh test -q
scripts/run.sh train
scripts/run.sh evaluate
scripts/run.sh predict data/genres_original/classical/classical.00000.flac
scripts/run.sh app
```

`train` runs eight main experiments and two bounded mel-CNN tuning trials. It uses at most 50 epochs, early stopping with patience 8, restores the best validation-loss weights, and selects the demonstration CNN using validation macro-F1. To run one experiment, use `scripts/run.sh train --models cnn_mel`. Model files are reused unless `--force` is explicit. Test data are loaded only by the separate evaluation stage. Final test evaluation refuses to silently replace results after model changes.

The environment includes the matching CUDA runtime, cuBLAS, cuDNN and companion NVIDIA pip libraries. `scripts/run.sh` adds their library directories to the dynamic linker path. It can also discover compatible existing laptop libraries as a fallback. GPU support was verified using both a convolution and its gradient. No system NVIDIA driver change was necessary.

## Dataset and methodology

The downloader streams the original GTZAN archive from the `marsyas/gtzan` mirror on Hugging Face, verifies its SHA-256, ignores AppleDouble metadata, and converts PCM audio to lossless FLAC. It does not retain an archive copy. The conversion receipt records decoding failures.

Preparation audits actual recordings, groups exact decoded-audio duplicates, excludes conflicting-label identical recordings, and uses 10 stratified group folds with seed 42: fold 0 is test, fold 1 validation, the remaining folds training. Split complete recordings before sampling windows. This prevents segments or exact duplicates from crossing partitions. It does not guarantee separation of artists or near-duplicate excerpts.

Training uses four seeded random two-second windows per recording. The main validation/test comparison uses one central window per recording. A separate five-window evaluation matches the demo. There are 44,100 waveform values or a 64-by-173 log-mel spectrogram per example. Mel parameters: 22,050 Hz mono, FFT/window length 512, hop 256, 64 bands, power 2, centered constant padding, decibel reference 1.0 and 80 dB range. No DCT is applied. Raw windows use per-window standardization. Mel statistics, classical scalers, and PCA are fitted only on training data.

Classical models use StandardScaler and PCA(15), then RBF SVM(C=1, gamma=scale) or k-NN(k=10, distance weighting). The FFNN has hidden widths 1024,512,256,128,64. CNN blocks use 32,64,128 filters, batch normalization, ReLU, pooling, dropout 0.25, and L2 1e-4, followed by dense widths 256,64,10. Raw CNN uses Conv1D; mel CNN uses Conv2D. All neural models use Adam, sparse cross-entropy, float32 and GPU training. JIT compilation is disabled for reproducibility and to avoid an additional XLA toolchain requirement.

## Files

- `src/`: preparation, features, model definitions, GPU check, training, evaluation, shared prediction.
- `app.py`: local upload demo.
- `tests/`: meaningful checks for leakage, stale caches, feature consistency, and audio edge cases.
- `data/cache/`: inventory, frozen splits, window provenance, feature arrays and fitted statistics.
- `results/`: model settings, progress, metrics, predictions, figures, GPU evidence, and selection metadata.
- `results/models/`: saved inference models, omitted from source Git history.
- `deliverables/`: generated reports, presentation, and submission bundle.
- `team.json`: editable names, IDs, section, repository URL, and faculty access details.

## Deliverable generation

The report builder uses the Codex bundled Python runtime (ReportLab). The slide builder uses the bundled Node runtime and `@oai/artifact-tool`. These authoring dependencies are not required to train or run the app. Generated documents read actual `results/evaluation.json`, so they never substitute published paper results for local results. See `docs/DELIVERY.md` and `docs/DEMO_AND_VIVA.md`.

## Interpretation and limitations

GTZAN is small, contains known repetitions and labeling faults, and reflects a limited set of historical genre labels. A track-disjoint split is stronger than a random segment split but cannot establish generalization to unseen artists. Single two-second windows omit long-term musical structure. The model forces one of ten labels and is not an open-set music detector. Softmax scores may be overconfident. Report actual validation/test results and failure cases. Do not claim human-level performance based on the source paper.

## References

- Huang, Serafini and Pugh. *Music Genre Classification*. Stanford CS229 report and poster supplied with the assignment.
- Tzanetakis and Cook. *Musical Genre Classification of Audio Signals*. IEEE Transactions on Speech and Audio Processing, 2002.
- Sturm. *The GTZAN dataset: Its contents, its faults, their effects on evaluation, and its future use*. https://arxiv.org/abs/1306.1461
- Dataset mirror: https://huggingface.co/datasets/marsyas/gtzan
- Librosa: https://librosa.org/doc/latest/
- TensorFlow GPU installation: https://www.tensorflow.org/install/pip
- Scikit-learn: https://scikit-learn.org/stable/
- Streamlit: https://docs.streamlit.io/

The implementation extends the starter code present in this directory. AI assistance was used for implementation and document preparation. Team members should review and understand the code, disclose assistance according to course policy, and record their actual contributions.

GPU floating-point kernels and library versions may introduce small differences across repeated runs or machines. Seeds fix the data split, window sampling, and initialization, but bitwise-identical GPU training is not guaranteed.
