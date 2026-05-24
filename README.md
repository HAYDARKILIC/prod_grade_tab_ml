<div align="center">

# Production-Grade Tabular Machine Learning

*A six-week curriculum that derives modern gradient boosting from first principles and rebuilds XGBoost, LightGBM, and CatBoost in pure NumPy before benchmarking, tuning, and ensembling them.*

</div>

---

## Curriculum

| Week | Theory | Practice |
|:----:|--------|----------|
| **1** | From AdaBoost to gradient boosting. Taylor expansion of the loss; first- and second-order derivatives ($g_i$, $h_i$). Tree-structure scoring and regularization ($\gamma$, $\lambda$). | Build a regression gradient booster from scratch — pure Python and NumPy, no scikit-learn. |
| **2** | XGBoost internals: sparsity-aware split finding (missing-value handling), weighted quantile sketch, parallel column scans, cache-aware access. | Analyze the learned default direction on a high-missingness dataset and visualize the trained trees. |
| **3** | Beyond XGBoost: GOSS (gradient-based one-side sampling), EFB (exclusive feature bundling), leaf-wise vs. level-wise growth and its overfitting risk. | Benchmark XGBoost vs. LightGBM on identical data — training time, memory, accuracy. |
| **4** | CatBoost: target-based encoding and target leakage. Ordered boosting as an unbiased gradient estimator. Symmetric trees and prediction-time speed. | High-cardinality categorical problem — replace manual encoding with CatBoost's native handling. |
| **5** | Why grid search wastes evaluations. Bayesian optimization and the Tree-structured Parzen Estimator (TPE). GPU (CUDA) execution of tree algorithms. | Optuna study that concurrently tunes all three libraries on GPU with median pruning. |
| **6** | — | End-to-end pipeline: XGBoost + LightGBM + CatBoost, tuned by Optuna, blended via weighted ensemble / stacking. |

---

## Repository

```
notebooks/         one notebook per week
src/               from-scratch implementations + production wrappers
scripts/           CLI entry points, YAML configs, benchmarks
tests/             pytest suite (29 tests passing)
```

---

## Key features

- **From-scratch implementations** of every algorithm in the lecture — second-order tree learner, sparsity-aware split finder, weighted quantile sketch, GOSS, EFB, ordered target statistics.
- **Production wrappers** around XGBoost / LightGBM / CatBoost with unified CPU/GPU API.
- **Optuna integration** with TPE sampling, median and Hyperband pruning, and persistent SQLite-backed studies.
- **Weighted ensembling and out-of-fold stacking** with leak-free meta-feature generation.
- **CI-ready engineering**: type hints, `mypy --strict`, `black`, `ruff`, pre-commit hooks, GitHub Actions.

---

## Installation

```bash
git clone https://github.com/HAYDARKILIC/prod_grade_tab_ml.git
cd prod_grade_tab_ml
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

GPU dependencies are documented in `environment.yml` (Conda).

---

## Usage

```python
from prod_grade_tab_ml.boosting import GradientBooster, SquaredError

# X, y: numpy arrays
model = GradientBooster(
    loss=SquaredError(), n_estimators=500,
    learning_rate=0.05, max_depth=6, reg_lambda=1.0,
).fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=50)
```

End-to-end capstone pipeline:

```bash
python scripts/run_pipeline.py --config scripts/configs/pipeline.yaml
```

Optuna study on a single algorithm:

```bash
python scripts/run_optuna_study.py --algorithm xgboost --n-trials 200 --gpu
```

---

## References

- Hastie, Tibshirani & Friedman — *The Elements of Statistical Learning* (2nd ed., 2009)
- Wade — *Hands-On Gradient Boosting with XGBoost and Scikit-Learn* (2020)
- Chen & Guestrin — *XGBoost: A Scalable Tree Boosting System* (KDD 2016)
- Ke et al. — *LightGBM: A Highly Efficient Gradient Boosting Decision Tree* (NeurIPS 2017)
- Prokhorenkova et al. — *CatBoost: Unbiased Boosting with Categorical Features* (NeurIPS 2018)
- Akiba et al. — *Optuna: A Next-Generation Hyperparameter Optimization Framework* (KDD 2019)

---

<sub>MIT-licensed. See [LICENSE](LICENSE).</sub>
