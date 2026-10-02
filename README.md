# Acoustic Species Identification in the Pantanal

Classical machine learning approach for the [BirdCLEF+ 2026](https://www.kaggle.com/competitions/birdclef-2026) competition, hosted by [Cornell Lab of Ornithology](https://www.birds.cornell.edu/), [TU Chemnitz](https://www.tu-chemnitz.de/), and [Google DeepMind](https://deepmind.google/).

This project was developed as part of the Machine Learning course (M1 Informatique) at Sorbonne University, Paris.

## About the competition

BirdCLEF+ 2026 challenges participants to identify wildlife species from audio recordings collected across the Pantanal wetlands in Brazil — one of the world's largest tropical wetlands, home to over 650 bird species and countless other animals. The dataset includes birds, amphibians, mammals, reptiles, and insects (234 species total, 206 of them with training audio), recorded through a network of ~1,000 acoustic recorders.

The goal is to build models that automatically detect which species are vocalizing in 5-second audio segments, supporting evidence-based conservation decisions in this critical ecosystem.

## Approach

Since this was a university project, **neural networks were not allowed**. This constraint made the project more challenging but also more educational — instead of feeding spectrograms into a CNN, I had to understand what makes audio signals distinguishable and engineer features by hand.

### Pipeline

1. **Feature extraction** — Extracted 180 audio features per recording using `librosa`:
   - 13 MFCCs (spectral envelope)
   - Spectral centroid, bandwidth, rolloff
   - Zero-crossing rate, RMS energy
   - 12 chroma features
   - Each summarized with 6 statistics: mean, std, min, max, 25th/75th percentiles

2. **Model comparison** — Compared 5 classical ML models:
   - Random Forest
   - Logistic Regression
   - Linear SVM
   - LightGBM
   - k-NN

3. **Analysis** — Ablation studies, error analysis, domain shift evaluation, taxonomic performance breakdown

### Key results

| Model | Macro ROC-AUC | Notes |
|-------|--------------|-------|
| **Logistic Regression + metadata** | **0.922** | Audio features plus location, rating and source |
| Logistic Regression | 0.897 | Best on audio features alone |
| LightGBM (tuned) | 0.865 | Required careful tuning |
| Random Forest | 0.770 | Solid baseline |
| Linear SVM | 0.867 | Slow |
| k-NN | 0.639 | Weak with high-dim features |

### Key findings

- **Logistic Regression outperformed all models** — well-engineered features create a linearly separable space, making simple models surprisingly effective
- **MFCCs are the most important features** — ablation showed a 0.14 ROC-AUC drop without them, while all other feature groups contribute less than 0.002 each
- **LightGBM failed with defaults** but recovered (0.50 to 0.87) with lower learning rate and more trees
- **Four metadata features lifted ROC-AUC from 0.893 to 0.922** (location, recording rating, source), more than any change to the audio features
- **Significant domain shift** between clean training audio and noisy field soundscapes (25% accuracy vs 9% hit rate)

## Report

The full write-up (method, experiments, ablations and error analysis) is in [`report/report.pdf`](report/report.pdf).

## Project structure

```
project/
├── notebooks/
│   ├── 01_eda.ipynb                  # Exploratory data analysis
│   ├── 02_features.ipynb             # Feature extraction pipeline
│   ├── 03_baseline.ipynb             # Random Forest baseline
│   ├── 04_feature_experiments.ipynb  # Feature set comparison
│   ├── 05_models.ipynb               # Model comparison (5-fold CV)
│   ├── 06_tuning_and_analysis.ipynb  # Tuning, ablation, error analysis
│   ├── 07_metadata.ipynb             # Adding non-audio metadata features
│   └── helpers.py                    # Feature extraction functions
├── report/
│   ├── report.pdf                    # Full project report
│   └── report.tex
├── figures/                          # All plots and results
├── requirements.txt
└── README.md
```

## How to reproduce

```bash
# 1. Clone the repo
git clone https://github.com/sheharyarIshfaq/birdclef-2026-classical-ml.git
cd birdclef-2026-classical-ml

# 2. Download the dataset from Kaggle
# https://www.kaggle.com/competitions/birdclef-2026/data
# Place it in birdclef-2026/

# 3. Set up environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install ipykernel
python -m ipykernel install --user --name=birdclef

# 4. Run notebooks in order (01 → 07)
jupyter notebook notebooks/
```

Note: Feature extraction (notebook 02) takes ~2 hours on the full dataset.

## Tools used

- [librosa](https://librosa.org/) — audio feature extraction
- [scikit-learn](https://scikit-learn.org/) — classical ML models and evaluation
- [LightGBM](https://lightgbm.readthedocs.io/) — gradient boosting
- [matplotlib](https://matplotlib.org/) / [seaborn](https://seaborn.pydata.org/) — visualization

## Dataset

The dataset is provided by the [Cornell Lab of Ornithology](https://www.birds.cornell.edu/) through the [BirdCLEF+ 2026 Kaggle competition](https://www.kaggle.com/competitions/birdclef-2026). It includes ~35,500 training audio recordings and ~10,600 soundscape recordings from the Brazilian Pantanal, supported by the Bezos Earth Fund AI for Climate and Nature Grand Challenge.

## Author

**Sheharyar**
M1 Informatique, Sorbonne University, Paris
