"""
GeoClimate-FusionLLM Configuration
All hyperparameters, paths, and constants in one place.
"""
import os

# === PATHS ===
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, "data")
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
FIGURES_DIR = os.path.join(ROOT_DIR, "figures")

RAW_DATA_FILE = os.path.join(DATA_DIR, "baghdad_2019-2024.xlsx")
FEATURES_FILE = os.path.join(RESULTS_DIR, "features.csv")
RESULTS_FILE = os.path.join(RESULTS_DIR, "paper_results.json")

# === DATA ===
TARGET = "tempmax"
DATE_COL = "datetime"
TRAIN_SPLIT_RATIO = 0.80
MIN_LAG_DAYS = 30  # skip first N days for rolling features

# === FEATURE ENGINEERING ===
LAG_DAYS = [1, 2, 3, 5, 7, 14]
ROLLING_WINDOWS = [3, 5, 7, 14, 21, 30]
TREND_WINDOWS = [3, 5, 7, 14]
EWM_SPANS = [3, 5, 10, 20]
SEASONAL_HARMONICS = [1, 2, 3]
ROLLING_DETAIL_WINDOWS = [7, 14]
ROLLING_DETAIL_VARS = [
    "tempmin", "dew", "humidity", "sealevelpressure",
    "cloudcover", "solarradiation", "windspeed"
]
LAG_VARS = [
    "tempmax", "tempmin", "temp", "dew", "humidity",
    "sealevelpressure", "cloudcover", "solarradiation", "windspeed"
]
TODAY_VARS = [
    "tempmax", "tempmin", "temp", "feelslikemax", "feelslikemin",
    "feelslike", "dew", "humidity", "sealevelpressure", "cloudcover",
    "solarradiation", "windspeed", "windgust", "winddir", "uvindex",
    "precip", "visibility", "solarenergy"
]
CHANGE_VARS = [
    "tempmax", "sealevelpressure", "humidity",
    "cloudcover", "dew", "windspeed"
]
HOT_FLAG_THRESHOLD = 45.0  # °C
COLD_FLAG_THRESHOLD = 18.0  # °C

# === BASELINES ===
RIDGE_ALPHAS = "logspace(-2, 3, 30)"  # evaluated via np.logspace
RIDGE_CV_FOLDS = 3

# MLP-Deep (Temporal Transformer proxy)
MLP_DEEP = dict(
    hidden_layer_sizes=(256, 128, 64, 32), activation="relu",
    solver="adam", alpha=0.001, max_iter=400, early_stopping=True,
    validation_fraction=0.15, n_iter_no_change=20, batch_size=64
)
# MLP-Shallow (TCN proxy)
MLP_SHALLOW = dict(
    hidden_layer_sizes=(128, 128, 64), activation="relu",
    solver="adam", alpha=0.01, max_iter=300, early_stopping=True,
    validation_fraction=0.15
)
# GBRT (N-BEATS proxy)
GBRT_BASELINE = dict(
    n_estimators=400, max_depth=5, learning_rate=0.05,
    subsample=0.8, min_samples_leaf=10
)

# === MMWSTM-ADRAN+ ===
STREAM1_MMWSTM_DEEP = dict(
    max_iter=500, max_depth=6, learning_rate=0.03,
    min_samples_leaf=10, l2_regularization=0.3,
    max_bins=255, early_stopping=True, validation_fraction=0.1,
    n_iter_no_change=20
)
STREAM2_MMWSTM_WIDE = dict(
    max_iter=700, max_depth=4, learning_rate=0.025,
    min_samples_leaf=20, l2_regularization=0.8,
    early_stopping=True, validation_fraction=0.1,
    n_iter_no_change=30
)
STREAM3_ADRAN = dict(
    hidden_layer_sizes=(512, 256, 128, 64), activation="relu",
    solver="adam", alpha=0.0003, learning_rate="adaptive",
    learning_rate_init=0.001, max_iter=400, early_stopping=True,
    validation_fraction=0.1, n_iter_no_change=25, batch_size=32
)
STREAM4_EXTRATREES = dict(
    n_estimators=400, max_depth=14, min_samples_leaf=4,
    max_features=0.6, n_jobs=-1
)
STREAM5_RF = dict(
    n_estimators=400, max_depth=12, min_samples_leaf=5,
    max_features=0.7, n_jobs=-1
)

# Fusion
FUSION_OPTIMIZER = "Nelder-Mead"
FUSION_RESTARTS = 50
FUSION_VALIDATION_FRACTION = 0.20  # last 20% of training for weight optim

# Residual calibration
RESIDUAL_CALIBRATOR = dict(
    max_iter=150, max_depth=3, learning_rate=0.05,
    min_samples_leaf=25, l2_regularization=2.0
)

# Tail specialists
TAIL_WEIGHT = 8.0  # sample weight multiplier for extremes
TAIL_HOT_PERCENTILE = 85  # training percentile for hot weighting
TAIL_COLD_PERCENTILE = 15
TAIL_BLEND_MAX = 0.4  # maximum blending weight for specialists
TAIL_SPECIALIST = dict(
    max_iter=400, max_depth=5, learning_rate=0.03,
    min_samples_leaf=8, l2_regularization=0.2
)

# === EVALUATION ===
TEST_HOT_PERCENTILE = 95
TEST_COLD_PERCENTILE = 5
ROBUSTNESS_SEEDS = [42, 123, 777]

# === FIGURES ===
FIGURE_DPI = 200
FIGURE_FORMAT = "png"
