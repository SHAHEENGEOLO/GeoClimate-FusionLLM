# Methodology Notes

## Feature Engineering — Temporal Causality
All features respect strict temporal causality. "Today" features (`*_t0`) are observed at day t, used to predict t+1. Lag features (`*_L{k}`) use values from k days prior. Rolling features use windows ending at day t (inclusive). Anomaly uses expanding mean from **previous years only** to prevent leakage.

## MMWSTM-ADRAN+ Design Rationale
**Multi-stream diversity:** Single models have correlated errors. Combining HistGBRT (two depth variants), DNN, ExtraTrees, and RF reduces variance through complementary learning mechanisms.

**Nelder-Mead fusion:** Optimizes 5 blending weights with 50 Dirichlet restarts, avoiding grid-search combinatorial explosion.

**Tail specialists:** Standard MSE weights all observations equally. 8× sample-weight upweighting focuses specialist models on the operationally critical extreme events. Adaptive blending ensures specialists contribute only in extreme prediction regions.

## Statistical Testing
Diebold-Mariano (1995) test with Newey-West HAC correction for autocorrelated forecast errors. Null hypothesis: equal predictive accuracy.
