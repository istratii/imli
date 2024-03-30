import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from utils import Classes, data_get, load, save

_RANDOM_STATE = 42
_MAX_ITER = 5000


class DiscriminatorClassifier:
    class _Base:
        def __init__(self, X, y, classes, model):
            assert classes and Classes.UNKNOWN not in classes
            self._classes = classes
            self._model = model
            self._estimator = make_pipeline(StandardScaler(), self._model)
            self._initialize_dataset(X, y, model.random_state)

        def _adapt_dataset(self, X, y, splitter):
            pass

        def _initialize_dataset(self, X, y, random_state=_RANDOM_STATE):
            sss = StratifiedShuffleSplit(
                n_splits=1, random_state=random_state, test_size=0.2
            )
            self._adapt_dataset(X, y, sss)

        def fit(self):
            return self._estimator.fit(self._X_train, self._y_train)

        def predict(self, X):
            return self._estimator.predict(X)

        def score(self):
            return self._estimator.score(self._X_test, self._y_test)

        def optimize(self, scaler_params, model_params, **grid_kwargs):
            estimator_steps_names = list(self._estimator.named_steps)
            params = dict()
            for name, values in scaler_params.items():
                params[f"{estimator_steps_names[0]}__{name}"] = values
            for name, values in model_params.items():
                params[f"{estimator_steps_names[1]}__{name}"] = values
            grid_search = GridSearchCV(self._estimator, params, **grid_kwargs)
            grid_search.fit(self._X_train, self._y_train)
            best_score = grid_search.score(self._X_test, self._y_test)
            if best_score > self._estimator.score(self._X_test, self._y_test):
                self._estimator = grid_search.best_estimator_
            return self._estimator

    class _Discriminator(_Base):
        def _adapt_dataset(self, X, y, splitter):
            y_bin = np.isin(y, self._classes).astype(int)
            train_idx, test_idx = next(splitter.split(X, y_bin))
            self._X_train = X[train_idx]
            self._y_train = y_bin[train_idx]
            self._X_test = X[test_idx]
            self._y_test = y_bin[test_idx]

    class _Classifier(_Base):
        def _adapt_dataset(self, X, y, splitter):
            mask = np.isin(y, self._classes)
            X_bis = X[mask]
            y_bis = y[mask]
            train_idx, test_idx = next(splitter.split(X_bis, y_bis))
            self._X_train = X_bis[train_idx]
            self._y_train = y_bis[train_idx]
            self._X_test = X_bis[test_idx]
            self._y_test = y_bis[test_idx]

    def __init__(
        self,
        X,
        y,
        classes,
        discriminator_model=LinearSVC(
            random_state=_RANDOM_STATE, max_iter=_MAX_ITER, dual="auto"
        ),
        classifier_model=SGDClassifier(
            random_state=_RANDOM_STATE,
            shuffle=False,
            early_stopping=True,
            n_jobs=4,
        ),
    ):
        self._classes = classes
        self._discriminator = self._Discriminator(X, y, classes, discriminator_model)
        self._classifier = self._Classifier(X, y, classes, classifier_model)
        self._initialize_dataset(X, y)

    def _initialize_dataset(self, X, y):
        sss = StratifiedShuffleSplit(
            n_splits=1, random_state=_RANDOM_STATE, test_size=0.2
        )
        train_idx, test_idx = next(sss.split(X, y))
        self._X_train = X[train_idx]
        self._y_train = y[train_idx]
        self._X_test = X[test_idx]
        self._y_test = y[test_idx]

    def fit(self):
        self._discriminator.fit()
        self._classifier.fit()

    def predict(self, X):
        discriminator_y = self._discriminator.predict(X)
        mask = discriminator_y == 1
        X_bis = X[mask]
        classifier_y = self._classifier.predict(X_bis)
        discriminator_y[mask] = classifier_y
        return discriminator_y

    def score(self, X=None, y=None):
        assert not ((X is None) ^ (y is None))
        X = X if X is not None else self._X_test
        y = y if y is not None else self._y_test
        y_pred = self.predict(X)
        mask = ~np.isin(y, self._classes)
        y_bis = y.copy()
        y_bis[mask] = Classes.UNKNOWN
        return accuracy_score(y_bis, y_pred)

    @classmethod
    def name(cls, classes):
        class_name = DiscriminatorClassifier.__name__.lower()
        classes = ",".join(str(int(cls_)) for cls_ in classes)
        return f"{class_name}{classes}"

    def save(self):
        save(self, DiscriminatorClassifier.name(self._classes))

    @staticmethod
    def load(classes):
        return load(DiscriminatorClassifier.name(classes))


if __name__ == "__main__":
    X, y = data_get()
    classes = [
        Classes.GRASS_HEALTHY,
        Classes.GRASS_STRESSED,
        Classes.GRASS_SYNTHETIC,
        Classes.TREE,
    ]
    vegetation_classifier = DiscriminatorClassifier(X, y, classes)
    vegetation_classifier.fit()
    vegetation_classifier._discriminator.optimize(
        dict(),
        dict(
            C=(100, 150, 200),
            random_state=range(0, 100),
            max_iter=range(5500, 10500, 500),
        ),
        n_jobs=8,
    )
    vegetation_classifier.save()
