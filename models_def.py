"""Model definitions used by train.py and the notebook."""
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import config


def _pipe(model, scale=True):
    steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", model))
    return Pipeline(steps)


def get_models():
    return {
        "KNN Regressor": _pipe(KNeighborsRegressor(n_neighbors=5)),
        "Linear Regression": _pipe(LinearRegression()),
        "Random Forest": _pipe(
            RandomForestRegressor(n_estimators=200, n_jobs=-1,
                                  random_state=config.RANDOM_STATE), scale=False),
        "Gradient Boosting": _pipe(
            GradientBoostingRegressor(random_state=config.RANDOM_STATE), scale=False),
    }


def tune_knn(X_train, y_train):
    """Find the best k for KNN with time-ordered cross-validation."""
    grid = GridSearchCV(
        _pipe(KNeighborsRegressor()),
        {"model__n_neighbors": [3, 5, 7, 11, 15, 21],
         "model__weights": ["uniform", "distance"]},
        cv=TimeSeriesSplit(n_splits=config.CV_SPLITS),
        scoring="neg_root_mean_squared_error", n_jobs=-1,
    )
    grid.fit(X_train, y_train)
    return grid
