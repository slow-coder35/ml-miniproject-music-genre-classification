# Measured experiment results

Main comparison: one central two-second window per held-out recording. There are 100 test recordings. Each training recording contributes four random windows. Model selection uses validation data.

| Experiment | Train accuracy | Validation accuracy | Test accuracy | Test macro-F1 |
|---|---:|---:|---:|---:|
| svm_mel | 56.0% | 46.0% | 46.0% | 0.430 |
| knn_mel | 100.0% | 43.0% | 45.0% | 0.430 |
| cnn_mel | 77.4% | 59.0% | 54.0% | 0.543 |
| ffnn_mel | 46.2% | 44.0% | 36.0% | 0.321 |
| svm_raw | 46.9% | 16.0% | 7.0% | 0.026 |
| knn_raw | 100.0% | 13.0% | 15.0% | 0.125 |
| ffnn_raw | 24.6% | 10.0% | 11.0% | 0.042 |
| cnn_raw | 27.1% | 20.0% | 21.0% | 0.149 |
| cnn_mel_lr0003 | 86.2% | 65.0% | 64.0% | 0.636 |
| cnn_mel_dropout040 | 78.6% | 60.0% | 54.0% | 0.539 |

The validation-selected demo model is **cnn_mel_lr0003**. Its separately evaluated five-window method has **70.0%** test accuracy and **0.706** macro-F1. This uses more audio than the central-window comparison.

Raw waveform models perform poorly on this split. The raw FFNN overfits rapidly, while the raw CNN learns slowly and achieves weak generalization. The tuned mel CNN improves on the baseline CNN. These are limited results on a small dataset with possible artist overlap, not a reproduction of the source paper's reported accuracy.

`training_run_seconds` in the CSV includes fitting, scoring, and model serialization. Neural histories show per-epoch training and validation behavior. Prediction files reproduce each metric exactly. The freeze record includes SHA-256 hashes for all ten saved model files.
