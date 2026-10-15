# Live demonstration and viva guide

## Five-minute demonstration

1. Explain that the objective is supervised prediction of one of ten music genres.
2. Show the saved comparison table and distinguish the central-window comparison from the five-window app protocol.
3. Start `scripts/run.sh app`. Upload a held-out track and play a few seconds.
4. Explain the waveform and mel spectrogram. Run prediction and discuss the top three scores.
5. Demonstrate a second contrasting genre and one misclassification. Avoid selecting examples solely because they succeed.
6. Upload a silent or corrupt file to show input handling.
7. Explain the principal limitation and conclude with the measured results.

## Questions to prepare

**Why log-mel features?** They represent frequency energy over time using perceptually spaced bands. A CNN can learn local time-frequency patterns. The experiment, rather than assumption, determines whether this improves results.

**Is this MFCC?** No. We use log-mel power directly. MFCC extraction additionally applies a cosine transform.

**Why split before windowing?** Windows from one recording share acoustic content. Crossing partitions would make test examples partly recognizable from training and overstate generalization.

**Why group duplicates?** Different filenames may contain identical audio. We group their decoded-audio hashes so they cannot cross partitions. Artist and near-duplicate leakage can still remain.

**Why PCA only for classical models?** Flattened raw/mel inputs are high dimensional. PCA reduces them to 15 training-derived components before distance/kernel classification. CNNs retain the input's structure.

**How does k-NN differ from SVM?** k-NN predicts using nearby training examples in PCA space. RBF SVM learns nonlinear class boundaries using support vectors.

**Why Conv1D for raw and Conv2D for mel?** Raw amplitude has a time axis. Mel features have frequency and time axes.

**What prevents overfitting?** Small data still leads to overfitting. CNN dropout, L2, batch normalization and validation early stopping mitigate it. Explain the actual training/validation curves.

**Why macro-F1?** It gives each genre equal weight and reflects both precision and recall. Accuracy is also reported for comparison with the source paper.

**How is the demo model selected?** Validation macro-F1 across three mel-CNN candidates, breaking ties using validation loss. The test set does not select it.

**Why can the app differ from the table?** The main comparison uses a fixed two-second central window. The app averages model scores over five windows, evaluated separately at the recording level.

**How do you know the GPU was used?** `results/gpu_check.json` records a GPU convolution and gradient. Neural run metadata records the visible GPU, and training utilization can be checked with NVIDIA tools.

**Can it classify any music correctly?** No. It always chooses among ten GTZAN labels. Scores are uncalibrated model preferences, and unfamiliar styles or mixed genres may fail.

**What did each person contribute?** Answer using the actual completed contribution record and explain code directly.
