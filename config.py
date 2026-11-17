"""Project settings. Leave a value as None to let the code auto-detect it."""

DATA_DIR = "data"          # folder that holds the Kaggle CSV file(s)
CSV_FILE = None            # e.g. "Plant_1.csv" (None = first CSV in data/)
TARGET_COL = None          # e.g. "Power" (None = auto-detect the solar power column)
TIME_COL = None            # e.g. "Time"  (None = auto-detect the timestamp column)

TEST_SIZE = 0.20           # last 20% of the timeline is used as the test set
RANDOM_STATE = 42
CV_SPLITS = 3              # TimeSeriesSplit folds for KNN tuning

MODEL_PATH = "models/best_model.joblib"
RESULTS_DIR = "results"
EXCEL_PATH = "solar_power_analysis_excel.xlsx"
