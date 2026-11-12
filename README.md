# Multi-Pathway CNN-BiLSTM for Daily Precipitation Prediction Under Resource Constraints

Code accompanying the paper **"A Novel Multi-Pathway CNN-BiLSTM Framework for Enhanced Precipitation Prediction Under Resource Constraints"** (Bouhouch, Riadsolh, Ez-Zahraouy; LaMCsI, Mohammed V University in Rabat).

A three-pathway CNN-BiLSTM with temporal attention, combined with a two-stage hurdle model (XGBoost rain-occurrence classifier + CNN-BiLSTM rainfall-amount regressor), for daily rainfall in the Australian Bureau of Meteorology "Rain in Australia" dataset. It trains in about 7 minutes per seed on a free Google Colab T4 GPU, within 12 GB of RAM.


---

## 1. Results (test set 2016-2017, original mm scale)

| Model | RMSE (mm) | MAE (mm) | R² |
|---|---|---|---|
| Naive Mean | 8.193 | 2.605 | -0.045 |
| Linear Regression | 7.695 | 2.263 | 0.078 |
| Random Forest | 7.520 | 2.201 | 0.119 |
| XGBoost | 7.343 | 2.134 | 0.160 |
| LightGBM | 7.346 | 2.132 | 0.160 |
| **CNN-BiLSTM** (mean ± std, 7 seeds) | 7.475 ± 0.067 | **2.093 ± 0.011** | 0.130 ± 0.016 |
| Hybrid (XGBoost + CNN-BiLSTM) | 7.427 | 2.146 | 0.141 |

All models degrade strongly on extreme events (≥ 10 mm); see the paper for the threshold analysis.

---

## 2. Quick start (Google Colab)

| Step | Action |
|---|---|
| 1 | Open the notebook: **https://colab.research.google.com/drive/1_B-EvVbgOI2CfvRyf5UJwKhuk_3CHnNB#scrollTo=Vh2gS0zuwf4A** |
| 2 | Set the runtime to **GPU (T4)**: Runtime → Change runtime type |
| 3 | Run the first cell to install dependencies |
| 4 | Upload the dataset file (see section 3) or mount your Drive |
| 5 | Run all cells. Seven seeds take about 49 minutes in total |

Local run: **[TODO: python version]**, then

```bash
pip install -r requirements.txt
python run_pipeline.py --data data/weatherAUS.csv --seeds 42 7 123 0 17 99 2024
```

**[TODO: replace with your real entry point and arguments]**

---

## 3. Data

| Item | Detail |
|---|---|
| Source | Australian Bureau of Meteorology daily observations ("Rain in Australia") |
| Size | 145,460 daily records, 49 stations, 2007-2017, 23 attributes |
| File | **weatherAUS.csv** |
| Where to get it | http://kaggle.com/datasets/jsphyg/weather-dataset-rattle-package

---

## 4. Pipeline (which step does what)

| Step | What it does | Key settings |
|---|---|---|
| 1. Imputation | Station-wise 7-day rolling mean, then forward/back fill, then zero fallback | Features cast to float32 |
| 2. Scaling | RobustScaler fit on training data only; target uses log1p | Applied to features and target |
| 3. Features | 78 variables per timestep: trends (temp, pressure, humidity), TempRange, lags 1/2/3/7, rolling mean/std 3/7/14, sin/cos month and day-of-year, location-mean rainfall | Lags use shift(1) before rolling to avoid leakage |
| 4. Sequences | 30-day windows + 8-dim learnable station embedding = 86 features per timestep | 49 stations |
| 5. Split | Strict chronological | Train 2007-2014, validation 2015, test 2016-2017 |
| 6. Stage 1 | XGBoost classifier on the last 7 days (546 features) | 300 estimators, depth 5, lr 0.05, subsample 0.8, early stopping on AUC |
| 7. Stage 2 | CNN-BiLSTM regressor, rainy days weighted 5× | See section 5 |
| 8. Prediction | Rain probability × predicted amount, clipped to [0, ∞) | |
| 9. Baselines | Naive Mean, Linear Regression, Random Forest (100 trees, depth 10), XGBoost and LightGBM (500 estimators) | Same scaler and target transform; last 7 days flattened |
| 10. Hybrid | Use CNN-BiLSTM when P(rain) > 0.5 and XGBoost prediction > 5 mm, else XGBoost | Applies to about 4.8% of test days |
| 11. Evaluation | RMSE, MAE, R² on the mm scale; thresholds at 0/5/10/20/30 mm; SHAP on 2,000 test sequences | |

---

## 5. Model and training

| Component | Detail |
|---|---|
| Pathway 1 | Causal Conv1D (64 filters, k=3) with residual skip, then BiLSTM (64 and 32 units), then temporal attention |
| Pathway 2 | Last 5 days (Cropping1D), flatten (430), Dense 64 then 32 |
| Pathway 3 | Global mean, global max and mean absolute deviation over 30 steps (258), Dense 64 then 32 |
| Head | Concatenate (128), Dense 128, 64, 32, then 1 (linear), with batch norm and dropout |
| Size | 203,378 parameters (202,546 trainable) |
| Optimiser | AdamW, lr 5×10⁻⁴, weight decay 10⁻⁴, gradient clip 1.0 |
| Loss | Huber |
| Schedule | Cosine annealing with 5-epoch linear warm-up |
| Callbacks | EarlyStopping (patience 15, restore best), ReduceLROnPlateau (factor 0.5, patience 7), TerminateOnNaN |
| Batch / epochs | 256 / max 120 |
| Seeds | 42, 7, 123, 0, 17, 99, 2024 |

---

## 6. Requirements

| Hardware used | Google Colab free tier, NVIDIA T4 GPU, 12 GB RAM (peak about 8.2 GB) |

---

## 7. Reproducibility notes

- Results are reported as the mean of 7 seeds with a 95% confidence interval (t-distribution). Single-seed numbers will differ slightly.
- GPU training is not bit-for-bit deterministic, so expect small differences even with the same seed.
- The best seed reaches R² = 0.146 (seed 2024).

---

## 8. Citation

```
Bouhouch F, Riadsolh A, Ez-Zahraouy H (2026) A novel multi-pathway CNN-BiLSTM
framework for enhanced precipitation prediction under resource constraints.
Theoretical and Applied Climatology. [TODO: status / DOI]

---

## 11. Contact

Corresponding author: Anouar Riadsolh, LaMCsI, Faculté des Sciences, Mohammed V University in Rabat, Morocco. **[TODO: e-mail]**
