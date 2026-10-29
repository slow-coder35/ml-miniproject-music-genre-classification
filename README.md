# Music Genre Classification — Project Guide

This README explains **what the project does, what has been implemented, where everything lives, how to run it, and how to explain it during the review**.

The project is located at:

```text
/home/bae/sem5/ml/miniproject/music_recommendation
```

The folder keeps the starter project's name, `music_recommendation`. The implemented task is **music genre classification**: give the system an audio recording and it predicts a genre.

## Contents

1. [Project at a glance](#1-project-at-a-glance)
2. [What is implemented](#2-what-is-implemented)
3. [Start the app or run a prediction](#3-start-the-app-or-run-a-prediction)
4. [How the complete pipeline works](#4-how-the-complete-pipeline-works)
5. [Folder and file guide](#5-folder-and-file-guide)
6. [Models and their configurations](#6-models-and-their-configurations)
7. [Actual experiment results](#7-actual-experiment-results)
8. [Reproduce the experiments](#8-reproduce-the-experiments)
9. [Reports, presentation, and submission](#9-reports-presentation-and-submission)
10. [Tests, troubleshooting, and limitations](#10-tests-troubleshooting-and-limitations)
11. [Suggested reading order and viva preparation](#11-suggested-reading-order-and-viva-preparation)
12. [Team and references](#12-team-and-references)

## 1. Project at a glance

| Item | What it means here |
|---|---|
| Input | A WAV or FLAC music recording containing at least two seconds of audio |
| Output | Predicted genre, the three highest model scores, and a score for each genre |
| Dataset | GTZAN music recordings, audited before training |
| Genres | Blues, classical, country, disco, hip-hop, jazz, metal, pop, reggae, rock |
| Models compared | SVM, k-nearest neighbors, feed-forward neural network, convolutional neural network |
| Input representations compared | Raw audio waveform and log-mel spectrogram |
| Experiments completed | Eight main comparisons and two additional mel-CNN tuning runs |
| Neural training hardware | Local NVIDIA RTX 4060 Laptop GPU |
| Demonstration | Local Streamlit app with upload, playback, waveform, spectrogram, and prediction |
| Model used by the app | `cnn_mel_lr0003`, selected using validation results |
| App-style test result | **70% accuracy and 0.706 macro-F1 on 100 held-out recordings** |

**Important:** the main experiment table uses one central two-second window per recording. The app averages scores from five windows. Its 70% result belongs to that separately evaluated five-window method.

## 2. What is implemented

| Component | Status | What was completed |
|---|---|---|
| Dataset acquisition | Implemented and run | Streamed the original archive from a mirror, checked its SHA-256, and saved lossless FLAC recordings |
| Dataset audit | Implemented and run | Checked decoding, duration, sampling rate, channels, and identical-audio hashes |
| Leakage prevention | Implemented and checked | Split complete recordings before windowing and kept exact duplicate groups together |
| Raw waveform features | Implemented and cached | Two-second windows with per-window normalization |
| Log-mel features | Implemented and cached | Spectrogram extraction and normalization using training statistics |
| Classical models | Trained and saved | SVM and k-NN, each on raw and mel inputs |
| Neural models | GPU-trained and saved | FFNN and CNN, each on raw and mel inputs |
| CNN tuning | Completed | Two additional trials varying learning rate or dropout |
| Final evaluation | Completed | Accuracy, macro-F1, per-class reports, confusion matrices, timing, and model sizes |
| Prediction evidence | Saved and verified | Per-example predictions reproduce the recorded metrics |
| Upload application | Implemented | Uses the saved model and the same feature extraction code as training |
| Automated checks | Passed | Nine functional tests, plus saved-model and portable-bundle checks |
| PDF summaries | Generated and visually checked | Two-page report and one-page alternative |
| Presentation | Generated and visually checked | Eight editable slides with measured results |
| Submission ZIP | Generated and checked | Source, documentation, results, reports, presentation, and the selected app model |
| Local Git repository | Created | Source and experiment evidence committed |
| Private GitHub publication | Pending | Saved GitHub authentication needs renewal |
| Faculty/TA repository access | Pending | Faculty/TA usernames and repository access must be supplied and verified |
| Section and contribution details | Partly complete | Names and IDs are filled; section and verified individual contributions remain editable |
| Visual browser review / demo recording | Outstanding | Browser access was denied during implementation; local application tests passed |

## 3. Start the app or run a prediction

### Option A: run the upload app

On this laptop, the environment and trained model already exist:

```bash
cd /home/bae/sem5/ml/miniproject/music_recommendation
scripts/run.sh app
```

Open the local address printed in the terminal, normally **http://127.0.0.1:8501**.

1. Upload a WAV or FLAC recording.
2. Listen to the audio and inspect the waveform and spectrogram.
3. Click **Predict genre**.
4. Read the predicted genre and the three highest model scores.

The app accepts files up to 20 MB and analyzes at most the first 30 seconds. It reports an error for unreadable, too-short, or silent recordings. Starting the app does not retrain any model.

You can also start it from the parent folder:

```bash
/home/bae/sem5/ml/miniproject/run_project.sh
```

If port 8501 is already in use, use the existing app or choose another port:

```bash
scripts/run.sh app --server.port 8502
```

### Option B: predict from the terminal

```bash
scripts/run.sh predict data/genres_original/classical/classical.00011.flac
```

The output is JSON containing the genre, model scores, model identifier, analyzed duration, and window positions. The same prediction code powers the app.

`results/demo_examples.json` lists example recordings and their actual saved predictions, including errors. For example, the saved rock example was predicted as country. These examples are useful for explaining limitations honestly.

### Option C: verify GPU availability

```bash
scripts/run.sh gpu-check
```

This runs a convolution and a gradient step on the GPU and saves evidence in `results/gpu_check.json`. Neural training and app inference require a working TensorFlow GPU. SVM, k-NN, PCA, and audio preprocessing run on the CPU.

## 4. How the complete pipeline works

```text
GTZAN archive
    |
    v
Download and lossless FLAC conversion
    |
    v
Audit recordings and identify exact duplicates
    |
    v
Split whole recordings into train / validation / test
    |
    v
Extract matching two-second raw and log-mel windows
    |
    v
Fit models using training data and monitor validation data
    |
    v
Choose the app CNN using validation macro-F1
    |
    v
Freeze model files and evaluate on the test partition
    |
    v
Save metrics, predictions, plots, and the model bundle
```

### Dataset preparation

GTZAN nominally has 1,000 recordings. One could not be decoded. The audit excluded two additional recordings whose identical audio had conflicting genre labels, leaving **997 usable recordings**.

The saved split is:

| Partition | Whole recordings | Windows used in the main comparison | Purpose |
|---|---:|---:|---|
| Training | 797 | 3,188 | Learn preprocessing statistics and model parameters |
| Validation | 100 | 100 | Monitor training and choose the demonstration checkpoint |
| Test | 100 | 100 | Measure performance after model choices are fixed |

The split uses ten stratified group folds and seed 42: fold 0 is test, fold 1 validation, and the other eight are training. Groups are based on exact decoded-audio hashes.

**Why split recordings first?** Clips from the same recording share musical content. If clips from one recording appear in both training and testing, the test result can overstate generalization.

### Feature extraction

Every recording is converted to mono at **22,050 samples per second**. A two-second window therefore contains **44,100 samples**.

- Training: four seeded random windows per recording.
- Main validation/test comparison: one fixed central window per recording.
- App: five evenly spaced windows, with duplicate positions removed for very short clips.

The raw and mel branches use matching source windows.

| Representation | Meaning | Shape per window |
|---|---|---|
| Raw waveform | Audio amplitude over time | `44,100` values |
| Log-mel spectrogram | Frequency energy over time, using mel-spaced frequency bands | `64 × 173` values |

Mel settings: FFT/window length 512, hop length 256, 64 mel bands, power 2, centered constant padding, decibel reference 1.0, and an 80 dB range. **No DCT is applied**, so these inputs are log-mel spectrograms rather than MFCCs.

Raw windows use their own mean and standard deviation. Mel normalization statistics are learned from training data only. The classical models' scaler and PCA are also fitted only on training data.

Features are cached so model training can reuse them. A signature links the cache to its split and feature settings. Changed data or settings cause a stale-cache error instead of silently reusing incompatible features.

### Training, selection, and evaluation

Training loads training and validation feature arrays. A separate evaluation program loads test arrays after the experiment choices are frozen.

Each neural run restores the checkpoint with the lowest validation loss. The app checkpoint is then chosen from the three mel-CNN candidates using **validation macro-F1**, with validation loss breaking ties.

Final evaluation saves model hashes, metrics, per-example predictions, confusion matrices, and learning curves. The five-window app method is evaluated separately from the central-window comparison.

## 5. Folder and file guide

Paths below are relative to the project folder.

```text
music_recommendation/
├── README.md                    # This guide
├── app.py                       # Streamlit upload interface
├── team.json                    # Names, IDs, section, and repository details
├── requirements.txt             # Pinned application / ML dependencies
├── requirements-gpu.txt         # Matching NVIDIA GPU libraries
├── requirements-dev.txt         # Development checks
├── environment-lock.txt         # Snapshot of the installed environment
├── pyproject.toml               # Formatting and test configuration
├── src/                         # Core ML and prediction implementation
├── scripts/                     # Launch, download, validation, and packaging helpers
├── tests/                       # Automated functional tests
├── data/                        # Local audio and feature caches
├── results/                     # Experiment evidence and trained models
├── docs/                        # Results explanation, delivery checklist, viva guide
├── deliverables/                # PDFs, presentation, and submission ZIP
└── work/                        # Original-code backup, logs, and private build files
```

### Core implementation: `src/`

| File | What it does | Main functions / classes to inspect |
|---|---|---|
| [`config.py`](src/config.py) | Defines paths, genre order, seed, feature settings, JSON helpers, and GPU library setup | `FEATURE_SPEC`, `initialize_gpu_environment()` |
| [`data.py`](src/data.py) | Loads audio, audits recordings, groups duplicates, and creates the recording split | `load_audio()`, `audit_dataset()`, `create_splits()`, `assert_disjoint()` |
| [`features.py`](src/features.py) | Extracts and caches raw and log-mel inputs; stores training mel statistics | `log_mel_window()`, `window_split()`, `build_feature_cache()`, `validate_cache_manifest()` |
| [`classical_models.py`](src/classical_models.py) | Builds the scaler → PCA → SVM/k-NN pipelines | `build_classical_model()` |
| [`nn_models.py`](src/nn_models.py) | Defines the FFNN, raw CNN, and mel CNN architectures | `build_ffnn()`, `build_cnn()` |
| [`train.py`](src/train.py) | Trains experiments, records run settings/history, and selects the demo checkpoint | `fit_experiment()`, `select_demo()` |
| [`evaluate.py`](src/evaluate.py) | Evaluates frozen models and exports metrics, prediction records, and figures | `main()`, `plot_confusion()` |
| [`predict.py`](src/predict.py) | Decodes uploaded audio and performs five-window saved-model inference | `decode_upload()`, `prediction_windows()`, `Predictor` |
| [`gpu_check.py`](src/gpu_check.py) | Checks GPU discovery and actual convolution/gradient execution | `configure_gpu()`, `main()` |

### Helper programs: `scripts/`

| File | Purpose |
|---|---|
| `run.sh` | Main launcher. Sets the project directory, GPU library paths, and thread limits |
| `download_dataset.py` | Streams and verifies the archive, ignores archive metadata, and writes lossless FLAC |
| `feature_figure.py` | Plots a real training waveform and its spectrogram |
| `validate_artifacts.py` | Checks model hashes, recomputes metrics from predictions, and compares independent model loads |
| `build_reports.py` | Generates both PDF summaries using measured results and `team.json` |
| `build_slides.mjs` | Generates and validates the editable eight-slide deck |
| `render_presentation.mjs` | Renders the exported deck for visual inspection |
| `package_submission.py` | Builds the submission ZIP and its file-checksum manifest |

The report and slide authoring helpers use the installed Codex document runtimes and currently contain paths specific to this laptop. **Training and running the app do not need those authoring runtimes.**

### Important saved files

| File / folder | What to find there |
|---|---|
| `data/download.json` | Download source, archive checksum, and decoding failures |
| `data/genres_original/` | Downloaded recordings stored as FLAC |
| `data/cache/inventory.json` | Per-recording audit information and usable/excluded status |
| `data/cache/splits.json` | Exact train, validation, and test recording lists |
| `data/cache/windows.json` | Parent recording, start sample, and label for every window |
| `data/cache/raw_*.npz` | Cached raw waveform features |
| `data/cache/mel_*.npz` | Cached normalized log-mel features |
| `data/cache/mel_stats.npz` | Training-derived mel mean and standard deviation |
| `results/audit/` | Copies of audit and provenance files suitable for the source repository |
| `results/models/` | All ten saved experiment models on this laptop |
| `results/*_run.json` | Settings, training/validation metrics, duration, model size, and GPU information |
| `results/*_history.json` | Per-epoch neural training and validation accuracy/loss |
| `results/results.csv` | Compact results table for all ten experiments |
| `results/evaluation.json` | Detailed per-class reports and five-window evaluation results |
| `results/evaluation_freeze.json` | Hashes of the evaluated models and the frozen demo selection |
| `results/predictions/` | Per-example predictions from which the metrics can be recomputed |
| `results/figures/` | Confusion matrices, learning curves, comparison charts, and class counts |
| `results/demo_model.json` | Selected model path, genre order, feature settings, and normalization statistics |
| `results/demo_examples.json` | Example recordings with their saved predictions, including mistakes |
| `results/validation.json` | Metric and independent model-load verification |
| `results/portable_validation.json` | Evidence that a separately extracted bundle could run prediction |

**The app needs the selected `.keras` model plus `results/demo_model.json`.** The selection file contains the training normalization statistics, so saved-model prediction does not need the full dataset or feature cache.

## 6. Models and their configurations

| Model | Plain-language explanation | Implemented configuration |
|---|---|---|
| SVM | Learns boundaries between genre examples in a reduced feature space | StandardScaler, PCA with 15 components, RBF SVM, `C=1`, `gamma="scale"` |
| k-NN | Looks at nearby training examples and combines their labels | StandardScaler, PCA with 15 components, ten neighbors, distance weighting |
| FFNN | Learns a sequence of transformations of the flattened input | Hidden widths 1024, 512, 256, 128, 64; ReLU; ten-class softmax |
| CNN | Learns local patterns in time or time-frequency inputs | Three convolution blocks with 32, 64, 128 filters; dense layers 256, 64, 10 |

Each model family has a raw-input and a mel-input experiment. The raw CNN uses one-dimensional convolutions. The mel CNN uses two-dimensional convolutions.

CNN blocks include batch normalization, ReLU, pooling, dropout 0.25, and convolution L2 regularization of `0.0001`. Mel kernels are 4×4, 3×3, 3×3 with 2×2 pooling. Raw kernels are 4, 3, 3 with pooling factor 8.

Neural training defaults:

- Adam optimizer, learning rate `0.001`.
- Sparse categorical cross-entropy loss.
- Batch size 32, automatically reduced on a GPU memory-exhaustion error.
- At most 50 epochs.
- Early stopping on validation loss, patience 8, restoring the best weights.
- Float32 and GPU execution, with JIT compilation disabled.

Additional mel-CNN trials:

| Experiment | Change from the baseline |
|---|---|
| `cnn_mel_lr0003` | Learning rate `0.0003`, dropout `0.25` |
| `cnn_mel_dropout040` | Learning rate `0.001`, dropout `0.40` |

The app uses **`cnn_mel_lr0003`**, which won the validation-based selection.

## 7. Actual experiment results

These numbers come from local saved runs. They are not copied from the reference paper.

### One central two-second window per held-out recording

| Experiment | Train accuracy | Validation accuracy | Test accuracy | Test macro-F1 |
|---|---:|---:|---:|---:|
| `svm_mel` | 56.0% | 46.0% | 46.0% | 0.430 |
| `knn_mel` | 100.0% | 43.0% | 45.0% | 0.430 |
| `cnn_mel` | 77.4% | 59.0% | 54.0% | 0.543 |
| `ffnn_mel` | 46.2% | 44.0% | 36.0% | 0.321 |
| `svm_raw` | 46.9% | 16.0% | 7.0% | 0.026 |
| `knn_raw` | 100.0% | 13.0% | 15.0% | 0.125 |
| `ffnn_raw` | 24.6% | 10.0% | 11.0% | 0.042 |
| `cnn_raw` | 27.1% | 20.0% | 21.0% | 0.149 |
| `cnn_mel_lr0003` — tuning | 86.2% | 65.0% | 64.0% | 0.636 |
| `cnn_mel_dropout040` — tuning | 78.6% | 60.0% | 54.0% | 0.539 |

Training accuracy is measured on the augmented training windows using the saved checkpoint. It can differ from an epoch's logged training accuracy because early stopping restores an earlier checkpoint and inference disables dropout.

### Five-window method used by the app

| Partition | Accuracy | Macro-F1 | Recordings |
|---|---:|---:|---:|
| Validation | 72.0% | 0.721 | 100 |
| Test | **70.0%** | **0.706** | 100 |

**Accuracy** is the fraction of correct predictions. **Macro-F1** averages the F1 score of each genre, giving every genre equal weight.

What the results show:

- Mel inputs improve test accuracy for all four model families on this split.
- The lower-learning-rate mel CNN improves on the baseline mel CNN.
- Averaging five windows improves the selected CNN's recording-level result.
- Raw models generalize poorly here; high training accuracy alone does not establish a useful classifier.

See [`docs/RESULTS.md`](docs/RESULTS.md) for the results discussion. The CSV field `training_run_seconds` includes fitting, scoring, and model serialization rather than pure GPU training time.

## 8. Reproduce the experiments

### Fresh environment

Use Linux, Python 3.12, a working NVIDIA driver, and sufficient storage. Run these commands from the extracted/cloned project directory:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-gpu.txt
scripts/run.sh gpu-check
scripts/run.sh download
scripts/run.sh prepare
scripts/run.sh test -q
scripts/run.sh train
scripts/run.sh evaluate
scripts/run.sh app
```

`requirements-gpu.txt` includes the main dependencies and the matching NVIDIA libraries. `scripts/run.sh` sets library paths before TensorFlow starts. The current environment was verified using a real GPU convolution and gradient, and its dependency check passed.

### Command reference

| Command | What it does | Reuse behavior |
|---|---|---|
| `scripts/run.sh download` | Downloads and converts the dataset | Reuses a completed download when the saved recording count matches |
| `scripts/run.sh prepare` | Audits, splits, extracts, and caches features | Reuses matching saved data/cache metadata |
| `scripts/run.sh train` | Runs all eight baselines and both tuning trials | Reuses completed compatible models unless `--force` is specified |
| `scripts/run.sh evaluate` | Evaluates the frozen experiment set | Reuses an identical completed evaluation; changed models require explicit `--force` |
| `scripts/run.sh predict AUDIO` | Predicts a recording using the selected saved CNN | Performs inference only |
| `scripts/run.sh app` | Starts the upload interface | Performs inference only |
| `scripts/run.sh test -q` | Runs the functional test suite | Does not train models |

Run one experiment or just the two tuning trials:

```bash
scripts/run.sh train --models cnn_mel
scripts/run.sh train --models tuning
```

Specify training limits:

```bash
scripts/run.sh train --models cnn_mel --epochs 50 --batch-size 16
```

Those options apply when the model is actually trained. To replace an existing experiment deliberately:

```bash
scripts/run.sh train --models cnn_mel --force
```

To rebuild changed preparation data:

```bash
scripts/run.sh prepare --force
```

Rebuilding data or retraining can invalidate previous model/evaluation records. Keep a copy of the original evidence before starting a new experiment series. Use validation results to choose settings, then evaluate the final frozen models. Do not repeatedly tune using test scores.

### Dataset acquisition details

The downloader streams `genres.tar.gz` from the `marsyas/gtzan` Hugging Face mirror, verifies the archive SHA-256, ignores AppleDouble metadata entries, and converts the original audio to lossless PCM16 FLAC. It saves a receipt instead of retaining a second archive copy.

The full dataset and caches are present locally. The submission ZIP includes their audit/provenance records and the selected model, while full dataset acquisition remains available through the download script.

## 9. Reports, presentation, and submission

| Deliverable | Location | Use |
|---|---|---|
| Two-page report | [`deliverables/project_report_two_pages.pdf`](deliverables/project_report_two_pages.pdf) | Detailed project summary |
| One-page alternative | [`deliverables/project_summary_one_page.pdf`](deliverables/project_summary_one_page.pdf) | Condensed version if faculty requires one page |
| Presentation | [`deliverables/project_presentation.pptx`](deliverables/project_presentation.pptx) | Eight editable review slides |
| Submission bundle | [`deliverables/music_genre_submission.zip`](deliverables/music_genre_submission.zip) | Source, evidence, documents, and selected app model |
| File checksums | `deliverables/SHA256SUMS.json` | Verify the files included in the bundle |

The guidelines conflict on report length, so both versions were prepared. Confirm which version faculty expects.

The local project contains all ten experiment models. The submission ZIP packages **only the selected demonstration model** to keep it small. Full model families and experiment recipes remain in the source.

Useful submission documents:

- [`docs/DELIVERY.md`](docs/DELIVERY.md): submission and private-repository checklist.
- [`docs/CONTRIBUTIONS.md`](docs/CONTRIBUTIONS.md): actual work completed by each student.
- [`docs/DEMO_AND_VIVA.md`](docs/DEMO_AND_VIVA.md): demonstration sequence and likely questions.
- [`docs/RESULTS.md`](docs/RESULTS.md): measured result table and interpretation.

Before final submission:

1. Fill the section in `team.json` and record verified individual contributions.
2. Renew GitHub authentication, create/push a private repository, and add faculty/TAs.
3. Verify their access and fill the repository details in `team.json`.
4. Review the app visually and rehearse the presentation and live demonstration.
5. Rebuild affected documents after changing team information.
6. Submit through the faculty-designated portal using the revised deadline confirmed by faculty.

On this laptop, refresh the ZIP after documentation or source changes with:

```bash
.venv/bin/python scripts/package_submission.py
```

PDF creation uses ReportLab in the installed Codex Python runtime. Slide creation uses the installed Codex Node runtime and `@oai/artifact-tool`. Their scripts read real `results/evaluation.json` and `team.json`; they do not generate invented results.

## 10. Tests, troubleshooting, and limitations

### What was checked

The nine functional tests cover recording/duplicate overlap rejection, feature shapes and normalization, stereo/resampled audio, duration limits, silence/corrupt uploads, deterministic prediction windows, stale caches, exclusion of test arrays from training, app startup, upload/prediction display, and error display.

Application flow tests use a controlled predictor to check the interface logic. Separate checks load the real saved model, compare independently loaded probabilities, recompute metrics from prediction files, verify model hashes, and run prediction from a separately extracted bundle.

The PDFs and final exported slides were rendered and visually inspected. Browser access was denied during implementation, so an actual browser-based visual review and live-demo recording remain outstanding.

### Common issues

| Issue | What to check |
|---|---|
| `.venv/bin/python` is missing | Create the environment and install `requirements-gpu.txt` |
| TensorFlow cannot see the GPU | Run `nvidia-smi`, then `scripts/run.sh gpu-check`; launch through `run.sh` so the CUDA library paths are set |
| A model file is missing | Restore the selected model bundle, or prepare data and run training |
| Feature cache is missing | Run `scripts/run.sh download`, then `scripts/run.sh prepare` |
| Cache is stale | Review what changed and deliberately rebuild with `scripts/run.sh prepare --force` |
| Training appears to finish immediately | Existing model/run records are being reused; use `--force` only when you intend to retrain |
| Evaluation refuses changed models | Its saved freeze differs; review the new experiment and use explicit `--force` only for an intentional new evaluation |
| Audio cannot be classified | Check WAV/FLAC format, at least two seconds of music, readable samples, and sufficient volume |
| The app predicts the wrong genre | Inspect the other scores and the audio; the model makes real errors and forces a choice among ten labels |
| Port 8501 is occupied | Use the existing server or choose `--server.port 8502` |

### Limits of the current project

- GTZAN is small and has documented repetitions, labeling faults, and distortions.
- Exact-duplicate grouping does not guarantee artist-disjoint or near-duplicate-free evaluation.
- A two-second window misses longer musical structure.
- The classifier always selects one of ten genres; unfamiliar styles, speech, or mixed genres may receive unreliable labels.
- Model scores are uncalibrated and can be overconfident.
- With 100 test recordings, one additional correct prediction changes accuracy by one percentage point.
- The reference paper's reported accuracy uses a different experiment setup and is not a directly equivalent benchmark.
- GPU kernels and library versions can cause small differences across repeated runs or machines despite fixed seeds.

Extensions such as music recommendation, artist recognition, genre conversion, transformer models, and cloud hosting are outside the current implementation.

## 11. Suggested reading order and viva preparation

Read the implementation in this order:

1. **`config.py`** — understand the genre order, sample rate, window size, and shared settings.
2. **`data.py`** — understand the audit, duplicate groups, and why splitting happens first.
3. **`features.py`** — understand raw normalization, log-mel extraction, and train-only statistics.
4. **`classical_models.py` and `nn_models.py`** — understand the four model families.
5. **`train.py`** — follow fitting, early stopping, saved models, and validation-based selection.
6. **`evaluate.py`** — follow frozen testing, metric calculation, and prediction exports.
7. **`predict.py`** — follow uploaded audio through preprocessing and five-window averaging.
8. **`app.py`** — connect the interface controls to the shared predictor.

A short explanation you should be able to give:

> We classify GTZAN music into ten genres. We split whole recordings before extracting short clips to reduce leakage, compare raw and log-mel inputs across four model families, and select the demonstration CNN using validation data. The selected model reaches 64% accuracy on central two-second test windows. The app averages five windows and reaches 70% accuracy in a separate recording-level test. We save the split, preprocessing settings, models, and predictions so the results can be checked.

Be ready to explain why mel features help, why PCA is used for classical models, how the raw and mel CNNs differ, what early stopping restores, why test data must not choose settings, and which mistakes the model makes. The longer question-and-answer guide is in [`docs/DEMO_AND_VIVA.md`](docs/DEMO_AND_VIVA.md).

## 12. Team and references

**Course:** UE24CS352A — Machine Learning

| Team member | Student ID |
|---|---|
| Prajwal Deshpande | PES1UG24CS589 |
| Prajwal Ambanna | PES1UG24CS588 |

Names and IDs are stored in `team.json`. Section, repository URL, and faculty access details remain editable there.

References:

- Huang, Serafini and Pugh. *Music Genre Classification*. Stanford CS229 report and poster supplied with the assignment.
- Tzanetakis and Cook. *Musical Genre Classification of Audio Signals*. IEEE Transactions on Speech and Audio Processing, 2002.
- [Sturm: GTZAN contents, faults, and evaluation](https://arxiv.org/abs/1306.1461).
- [GTZAN dataset mirror](https://huggingface.co/datasets/marsyas/gtzan).
- [Librosa documentation](https://librosa.org/doc/latest/).
- [TensorFlow GPU installation](https://www.tensorflow.org/install/pip).
- [Scikit-learn documentation](https://scikit-learn.org/stable/).
- [Streamlit documentation](https://docs.streamlit.io/).

The implementation extends the starter code that was already present in this directory. AI assistance was used for implementation and document preparation. Team members should review the code, explain the experiments themselves, disclose assistance according to course policy, and record their actual contributions.
