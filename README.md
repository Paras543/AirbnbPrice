# NYC Airbnb Price Prediction

**Predicting nightly Airbnb listing prices in New York City with a PyTorch feed-forward neural network — from a raw 90-column scrape down to a trained regressor with an R² of 0.55.**

**Tech stack:** Python · Pandas · Scikit-learn · PyTorch · Matplotlib · Seaborn

- Cleaned and reduced a 30,259-row / 90-column raw Airbnb scrape down to 24 structurally meaningful features across three deliberate pruning passes
- Engineered features from messy raw fields — parsed JSON amenity lists into an `amenities_count` signal, cleaned currency-formatted price strings, one-hot encoded 49 property types and 4 room types
- Corrected severe right-skew in price via `log1p` transform, verified with pre/post histograms and a full 92-feature correlation heatmap
- Trained a 91 → 64 → 32 → 16 → 1 fully connected network (8,513 params) with Adam + MSE loss, evaluated on a held-out test split with MAE / RMSE / R² and a real prediction-vs-actual spot check

---

## Overview

Airbnb hosts in a dense market like NYC have to price competitively without much built-in
platform guidance. This project builds an end-to-end pipeline — cleaning, feature
engineering, exploratory analysis, and a neural network regressor — to predict nightly
listing price from structural, locational, and host-trust signals alone (no listing text
or images).

A written report of the full methodology and results is included as an HTML page —
see [`airbnb-price-prediction.html`](./airbnb-price-prediction.html).

## Dataset

- **Source:** Scraped NYC Airbnb `listings.csv` (Inside Airbnb–style format)
- **Size:** 30,259 listings × 90 raw columns
- **Target:** Nightly price (log-transformed for training)

## Data Cleaning & Feature Engineering

| Step | Description |
|---|---|
| Pass 1 | Dropped identifiers, scrape metadata, redundant host-listing-count aggregates → 90 → 54 columns |
| Pass 2 | Dropped high-missingness / low-signal host fields and post-hoc review timing fields → 54 → 34 columns |
| Pass 3 | Dropped free-text fields and duplicate geography columns → 34 → 24 columns |
| Encoding | One-hot encoded `property_type` (49 categories) and `room_type` (4 categories); label encoded `host_identity_verified`, `has_availability` |
| Amenities | Parsed JSON amenity lists → single `amenities_count` feature |
| Price cleaning | Stripped currency symbols/commas, coerced to numeric, `log1p` transformed |
| Imputation | Median fill for numeric columns, mode fill for categorical columns |

Final model input: **91 features** after encoding.

## Model

```
Input (91) → Linear(64) → ReLU → Linear(32) → ReLU → Linear(16) → ReLU → Linear(1)
```

- **Framework:** PyTorch
- **Loss:** MSE
- **Optimizer:** Adam (lr = 0.001)
- **Batch size:** 8
- **Epochs:** 10
- **Params:** 8,513 trainable

## Results

| Metric | Value |
|---|---|
| MAE (log-price) | 0.350 |
| RMSE (log-price) | 0.473 |
| R² | 0.549 |

The model explains just over half the variance in log-price using structural and
locational features alone. The strongest positive correlates of price were `accommodates`,
`beds`, and `bedrooms`; the strongest negative correlates were private-room listings and
longitude (distance from Manhattan's core).

## Limitations

- No text, image, or review-content features — only structural/tabular signals
- Uniform median/mode imputation, despite ~30% missingness in `bathrooms`, `bedrooms`, `beds`
- Single fixed train/val/test split — no k-fold cross-validation
- Rare `property_type` categories (e.g. "Cave", "Boat") are underrepresented

## Future Work

- TF–IDF or embedding features from listing descriptions
- Neighbourhood-level aggregate price features
- Benchmark against gradient-boosted trees (XGBoost, LightGBM)

## Setup

```bash
git clone https://github.com/<your-username>/nyc-airbnb-price-prediction.git
cd nyc-airbnb-price-prediction
pip install -r requirements.txt
```

**requirements.txt**
```
pandas
numpy
scikit-learn
torch
torchsummary
matplotlib
seaborn
```

## Project Structure

```
.
├── Airbnb.ipynb                    # Full EDA, preprocessing & model training notebook
├── airbnb-price-prediction.html    # Written report (website format)
├── README.md
└── requirements.txt
```

## Usage

Open `Airbnb.ipynb` in Jupyter or Google Colab, place `listings.csv` in the working
directory, and run all cells sequentially — the notebook covers cleaning, EDA, model
training, and evaluation end to end.

---

<p align="center"><i>Built as part of an ongoing ML/AI portfolio.</i></p>
