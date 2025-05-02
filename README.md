# Sonatus Coderbyte Assessment 

# APS Failure Detection – Brief Model Report

## 1 / Data & Pre‑processing
- **Files used**  
  - `aps_failure_training_set.csv` (60 000 rows)  
  - `aps_failure_test_set.csv` (16 000 rows)  
- **Cleaning steps**  
  1. Replace `"na"` with `NaN`.  
  2. Map labels: `neg → 0`, `pos → 1`.  
  3. **Median imputation** for every feature.  
  4. **Standard scaling** (keeps tree splits numerically stable).  

## 2 / Model
- **Algorithm:** XGBoost gradient‑boosted trees.  
- **Key hyper‑params**
  | param | value |
  |-------|-------|
  | n_estimators | 700 |
  | max_depth | 6 |
  | learning_rate | 0.05 |
  | subsample / colsample_bytree | 0.8 / 0.8 |
  | scale_pos_weight | 59 (matches class imbalance) |

- **Threshold tuning**  
  - Costs: `Cost₁ = 10` (FP), `Cost₂ = 500` (FN).  
  - Sweep probability cut‑off on 20 % validation split → **τ\*** ≈ 0.08 gives the lowest cost.

## 3 / Performance

| Split | FP | FN | Precision | Recall | **Total Cost** (`10·FP + 500·FN`) |
|-------|----|----|-----------|--------|----------------------------------|
| Validation (12 k rows) | 94 | 12 | 0.69 | 0.95 | **6 140** |
| Public Test (16 k rows) | 360 | 5 | 0.51 | 0.99 | **11 830** |

**Interpretation**
- Only **5 missed failures** on the test set (high recall ≈ 99 %).  
- False‑alarm rate is moderate; they cost less (10 pts each) so overall cost stays low.  
- Beats a naïve Gaussian‑NB baseline (15 090 cost) by ~25 %.

## 4 / Next easy gains
- Add “was‑missing” indicator columns to exploit sparsity.
- Grid‑search depth, trees, and learning rate (often pushes cost < 10 000).
- Test LightGBM or CatBoost; both handle sparse numeric data well.
- Use a cost‑sensitive custom objective or post‑train calibration for finer thresholding.
