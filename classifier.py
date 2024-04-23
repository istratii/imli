import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from utils import Classes


class Classifier:
    def __init__(self, classes, model, random_state=42):
        self._random_state = random_state
        self._classes = classes
        self._model = model
        self._estimator = make_pipeline(StandardScaler(), self._model)

    def fit(self, X, y):
        mask = np.isin(y, self._classes)
        y[~mask] = Classes.UNKNOWN
        return self._estimator.fit(X, y)

    def score(self, X, y):
        y_pred = self.predict(X)
        mask = np.isin(y, self._classes)
        y[~mask] = Classes.UNKNOWN
        return accuracy_score(y, y_pred)

    def predict(self, X):
        return self._estimator.predict(X)

    def _dataset_get(self, X, y, include_unknown=False, n_splits=1, test_size=0.2):
        splitter = StratifiedShuffleSplit(
            n_splits=n_splits, random_state=self._random_state, test_size=test_size
        )
        mask = np.isin(y, self._classes)
        if include_unknown:
            y[~mask] = Classes.UNKNOWN
        else:
            X, y = X[mask], y[mask]
        train_idx, test_idx = next(splitter.split(X, y))
        return X[train_idx], y[train_idx], X[test_idx], y[test_idx]

    def optimize(
        self,
        X,
        y,
        include_unknown=False,
        scaler_params=dict(),
        model_params=dict(),
        **grid_kwargs,
    ):
        estimator_steps_names = list(self._estimator.named_steps)
        assert len(estimator_steps_names) > 1
        params = dict()
        for name, values in scaler_params.items():
            params[f"{estimator_steps_names[0]}__{name}"] = values
        for name, values in model_params.items():
            params[f"{estimator_steps_names[1]}__{name}"] = values
        X_train, y_train, X_test, y_test = self._dataset_get(
            X, y, include_unknown=include_unknown
        )
        cv = GridSearchCV(self._estimator, params, **grid_kwargs)
        cv.fit(X_train, y_train)
        best_score = cv.score(X_test, y_test)
        if best_score > self._estimator.score(X_test, y_test):
            self._estimator = cv.best_estimator_
        return best_score

    def cross_validation(self, n_splits=1):
        pass
