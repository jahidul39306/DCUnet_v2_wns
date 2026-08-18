# DCUnet_v2_wns

**Deep Complex U-Net for Speech Enhancement and Wind Noise Suppression**

This repository contains our implementation of a **Deep Complex U-Net (DCUNet)** for speech enhancement, developed as a university project (P6, Computer Vision) during the **second semester of our Master's degree** at Aalborg University. The project was completed as a group project by **six students**.

The goal of the project was to investigate how deep learning can be used to suppress noise — and specifically outdoor wind noise — from speech recordings while preserving the quality and intelligibility of the original speech. Our approach operates in the **complex-valued Short-Time Fourier Transform (STFT) domain**, using a Deep Complex U-Net architecture to estimate a complex ratio mask (CRM) that is applied to the noisy spectrogram to produce an enhanced speech representation.

The project gave us practical experience with several areas of machine learning and audio processing, including:

* Speech enhancement and wind noise suppression
* Short-Time Fourier Transform (STFT) and complex ratio masking
* Complex-valued neural networks
* U-Net architectures
* PyTorch and torchaudio
* Audio dataset preparation, preprocessing, and custom real-world data collection
* Model training and checkpointing on an HPC cluster
* Speech-quality evaluation using metrics such as **STOI, PESQ, and SI-SDR** (intrusive) and Torch **SQUIM** (non-intrusive)
* Running inference on noisy speech and producing enhanced audio

## Project Structure

* [complex_model.py](complex_model.py) - Deep Complex U-Net model implementation (complex 2D convolutions/transposed convolutions and complex batch norm)
* [complex_data_prep.py](complex_data_prep.py) - Dataset loading, preprocessing (resampling, peak normalization, 4s segmenting), and STFT preparation
* [complex_train.py](complex_train.py) - Training loop, validation, checkpointing, and wSDR loss
* [complex_inference.py](complex_inference.py) - Batch model inference and evaluation across multiple models/testsets
* [complex_local_inference.py](complex_local_inference.py) - Runs a trained checkpoint on external `.wav` files for quick real-world checks
* [config.py](config.py) - Configuration options
* [convert_to_csv.py](convert_to_csv.py) - Converts evaluation metric results to CSV
* [torch_squim_evaluator.py](torch_squim_evaluator.py) - Non-intrusive metric computation (no clean reference available) using Torch SQUIM
* [submit_job.sh](submit_job.sh) - Slurm-based HPC job submission script (GPU allocation, environment setup)
* [requirements.txt](requirements.txt) - Python dependencies

## About the Project

The central idea behind the project is to represent speech in the frequency domain rather than processing the raw waveform directly. The input audio is transformed into a complex-valued STFT representation containing both magnitude and phase information.

The noisy representation is then passed through a **Deep Complex U-Net**, which learns to estimate a complex ratio mask. Applying the mask to the noisy spectrogram yields a cleaner representation of the speech, which is transformed back into the time domain (inverse STFT) to produce the enhanced audio.

This project was particularly useful for understanding the relationship between:

**audio signal processing → STFT → complex-valued representations → deep learning → speech/wind noise enhancement**

### Model variants

Two models were trained and compared to study how training data composition affects wind noise suppression performance:

* **Model A** — trained from scratch on our **custom outdoor wind-noise dataset**, recorded with dynamic and shotgun microphones (with windshields) at 20 cm and 50 cm from the speaker.
* **Model B** — trained on a subset of the **Microsoft DNS (Deep Noise Suppression) Challenge dataset**, clipped to 4-second segments, covering a broad range of synthetic/general noise types.

## Dataset

The project uses paired clean and noisy speech recordings. The expected dataset structure is:

```text
dataset/
├── clean/
│   ├── speech_001.wav
│   ├── speech_002.wav
│   └── ...
└── noisy/
    ├── speech_001.wav
    ├── speech_002.wav
    └── ...
```

Each clean recording should have a corresponding noisy recording. The dataset loader automatically resamples audio to the target sampling rate (16 kHz by default) and limits the maximum segment duration used during training (4 seconds).

Three evaluation sets were used in the report:

* **Testset A** — artificially mixed speech + noise at fixed SNR levels (-5 dB / +5 dB), with a clean reference available.
* **Testset B** — real-world recordings made outdoors with dynamic and shotgun microphones at 20 cm / 50 cm, with a clean reference available.
* **Testset C** — real, live two-speaker recordings in windy conditions with **no clean reference**, evaluated with the non-intrusive Torch SQUIM model.

Our custom wind-noise dataset is available [here](https://aaudk-my.sharepoint.com/:f:/g/personal/jw18bw_student_aau_dk/IgA5xBT1E5_0Qr-JpxdNFahnAeCEOwLG3wL6dT1OIIrh8rw?e=aySSbq), and example audio clips (noisy vs. enhanced) can be listened to on the [audio demo page](https://bgiri25.github.io/wns-audio-demo/).

## Model

The main model is a **Deep Complex U-Net (DCUNet)**. Unlike a conventional U-Net operating on real-valued images or spectrograms, the model operates on complex-valued STFT representations, using an encoder-decoder structure with skip connections and complex-valued convolutions/batch normalization to jointly exploit magnitude and phase information.

### Training setup

* **Hardware:** Nvidia L4 GPU (HPC cluster, Slurm job submission)
* **Epochs:** 120 for the general noise suppression baseline (Model B), 300 for the wind noise suppression model (Model A)
* **Batch size:** 16
* **Optimizer:** Adam, with a staged learning-rate schedule — an initial higher rate of `2e-4` to learn broad features, dropped to `5e-5` for fine-tuning; the wind-specific run used a fixed rate of `1e-4`
* **Loss:** weighted Signal-to-Distortion Ratio (**wSDR**) loss, computed in the time domain, balancing speech-target and noise-target SDR via a weighting factor α

Both models reached stable convergence within the first ~20-60 epochs and stayed near a wSDR loss of about **-10 dB to -12 dB** for the remainder of training, with no signs of instability or divergence at longer training horizons.

## Training

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Prepare the dataset in the expected structure and configure the training parameters in `complex_train.py`.

The main parameters include:

* `BATCH_SIZE`
* `NUM_EPOCHS`
* `LEARNING_RATE`
* `SAVE_DIR`

Start training with:

```bash
python complex_train.py
```

Training checkpoints and logs are stored in:

```text
complex_checkpoints/
├── latest_checkpoint.pth
├── best_model.pth
└── training_log.csv
```

Pretrained weights for both models are available for download:

* Model A (wind, [without overlapping segments](https://aaudk-my.sharepoint.com/:u:/g/personal/jw18bw_student_aau_dk/IQCbF7wp-FRiSowxtb7Olk1gATjfXInm1aJlVQ5lm2MBfwk?e=fJvv0l) / [with overlapping segments](https://aaudk-my.sharepoint.com/:u:/g/personal/jw18bw_student_aau_dk/IQDdSLaB-H5FT5__nfEONFD3AbuhDe0gjUnwr80EbfNSh-w?e=QvAvDd))
* [Model B (DNS)](https://aaudk-my.sharepoint.com/:u:/g/personal/jw18bw_student_aau_dk/IQB7Gz4VVIneQYmvdsBlq5uyARgh7OtTaTZaCLf7mdD31ac?e=o4Dgqx)

## Inference and Evaluation

The `complex_inference.py` script can be used to load a trained model, process noisy speech, generate enhanced audio, and evaluate the results across multiple models/testsets. `complex_local_inference.py` is the lighter single-checkpoint version for quick checks on external `.wav` files.

The evaluation pipeline supports metrics including:

* **STOI** - Short-Time Objective Intelligibility
* **PESQ** - Perceptual Evaluation of Speech Quality
* **SI-SDR** - Scale-Invariant Signal-to-Distortion Ratio
* **Torch SQUIM** (non-intrusive, reference-free versions of the above) — used for Testset C, where no clean reference exists

Before running inference, configure the test-set paths and model checkpoints in `complex_inference.py`.

Then run:

```bash
python complex_inference.py
```

Time alignment (via cross-correlation) is applied between enhanced and clean signals before metric computation, since the model can introduce a small time shift in the output waveform.

## Results

Model B (trained on the larger, more varied DNS dataset) consistently outperformed Model A (trained only on the custom wind dataset) on the intrusive metrics, across both the artificially mixed test set and the real-world recordings.

### Testset A — artificially mixed speech (SNR = -5 dB / +5 dB)

| Model | SNR | STOI (noisy → enhanced) | SI-SDR dB (noisy → enhanced) | PESQ (noisy → enhanced) |
|---|---|---|---|---|
| Model A (no overlap) | -5 dB | 0.612 → 0.632 | -4.99 → -1.09 | 1.24 → 1.51 |
| Model A (no overlap) | +5 dB | 0.784 → 0.752 | 5.00 → -0.18 | 1.71 → 1.81 |
| Model A (with overlap) | -5 dB | 0.612 → 0.642 | -4.99 → -1.02 | 1.24 → 1.57 |
| Model A (with overlap) | +5 dB | 0.784 → 0.758 | 5.00 → -0.96 | 1.71 → 1.93 |
| **Model B (DNS)** | -5 dB | 0.612 → **0.751** | -4.99 → **12.9** | 1.24 → **2.01** |
| **Model B (DNS)** | +5 dB | 0.784 → **0.854** | 5.00 → **17.8** | 1.72 → **2.59** |

### Testset B — real-world recordings, average across microphones/distances

| Model | STOI (noisy → enhanced) | SI-SDR dB (noisy → enhanced) | PESQ (noisy → enhanced) |
|---|---|---|---|
| Model A, no overlap | 0.867 → 0.856 | 1.428 → -2.825 | 1.891 → 2.125 |
| Model A, with overlap | 0.867 → 0.856 | 1.428 → -2.899 | 1.891 → 2.114 |
| **Model B (DNS)** | 0.867 → **0.876** | 1.428 → **6.852** | 1.891 → **2.305** |

### Testset C — real, windy, two-speaker recordings (non-intrusive / SQUIM, no clean reference)

| Microphone / condition | STOI est. (noisy → enhanced) | SI-SDR dB est. (noisy → enhanced) | PESQ est. (noisy → enhanced) |
|---|---|---|---|
| Dynamic (shield) | 0.974 → 0.870 | 5.25 → 6.79 | 1.66 → 1.61 |
| Dynamic (no shield) | 0.930 → 0.834 | -10.4 → 3.70 | 1.19 → 1.43 |
| Condenser (shield) | 0.824 → 0.799 | -18.4 → 0.496 | 1.35 → 1.29 |
| Phone | 0.671 → 0.643 | -14.4 → -8.83 | 1.19 → 1.16 |
| Shotgun (no shield) | 0.838 → 0.699 | -17.1 → -5.12 | 1.14 → 1.18 |

### Key findings

* **Model B (DNS-trained) generally scores higher** on the intrusive metrics — up to **+17.8 dB SI-SDR** and a PESQ of **2.59** on artificially mixed data — most likely because the DNS dataset is much larger and more acoustically varied than our custom wind recordings.
* Enhanced **shotgun microphone** recordings consistently scored better than enhanced **dynamic microphone** recordings on Testset B, an effect flagged for future investigation.
* **Model A performs better when two speakers overlap** in the recording (a scenario relevant to interviews), while Model B better preserves the speech frequency spectrum overall when overlap isn't an issue.
* Phase analysis showed that both models learn a small phase correction at low frequencies; Model A over-corrects phase in higher, otherwise-clean frequency ranges, while Model B leaves phase largely untouched — including where correction would help (e.g. the ~5 kHz phase degradation seen in noisy shotgun recordings).
* There is a practical trade-off: **Model B** is more aggressive and suppresses all background noise (better for maximum isolation), while **Model A** targets wind specifically and preserves more natural ambient sound (useful when some background presence is desirable, e.g. broadcast-style recording).

## What We Learned

This project was an opportunity to apply concepts from machine learning, signal processing, and deep learning to a real-world audio problem.

Some of the main concepts we worked with were:

1. How audio signals can be transformed from the time domain into the frequency domain using STFT.
2. How magnitude and phase information are represented using complex numbers.
3. How U-Net architectures can be adapted for speech enhancement.
4. How complex-valued neural networks differ from conventional real-valued networks.
5. How to prepare paired datasets for supervised speech enhancement, including collecting and labeling our own real-world wind-noise recordings.
6. How to train, validate, checkpoint, and evaluate a deep learning model, including on an HPC/Slurm cluster.
7. How objective metrics such as STOI, PESQ, and SI-SDR — and non-intrusive alternatives like Torch SQUIM — can be used to evaluate speech enhancement systems with and without a clean reference.
8. The practical challenges involved in training deep learning models on audio data, including time-alignment issues introduced by the model itself.

## Limitations

This project was developed primarily as a university learning project. It is not intended to be a production-ready speech enhancement system.

Some limitations include:

* Limited size of the custom wind-noise training dataset
* Fixed STFT and training parameters
* Limited experimentation with different architectures
* Evaluation dependent on the available datasets and metrics
* No real-time optimization or deployment pipeline
* Model B underperforms in overlapping two-speaker scenarios; the cause is not yet fully understood

## Future Work

Possible improvements, based on findings in the project report:

* Investigating why Model B performs poorly with overlapping speakers, and addressing it
* Improving real-time inference performance for deployment
* Experimenting with additional loss functions to improve phase reconstruction
* Collecting more and stronger (high-gust) wind-noise recordings to expand the custom dataset
* Testing the models on a wider range of real-world applications and devices
* Comparing DCUNet against real-valued U-Net and other, more recent speech enhancement architectures

## Acknowledgements

This project was completed as part of our Master's degree coursework (P6, Computer Vision) during the second semester, at Aalborg University.

The implementation was developed collaboratively by our group of six members. We also used AI-assisted development tools during the project for code exploration, debugging, documentation, and understanding implementation details.

A companion general noise suppression model (Model B) developed alongside this project is available at [UNet-V1](https://github.com/akshara-aau/UNet-V1).

## License

No explicit license is currently provided for this project.

**Note**: This README was generated with the assistance of an AI based on my experience and understanding of the project.