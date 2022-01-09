from typing import Any, Dict, Tuple

import numpy as np
from pydantic.typing import Literal
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted


class Winsorizer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        quantile_range: Tuple[float, float] = (0.1, 0.9),
        nan_policy: Literal["propagate", "raise", "omit"] = "propagate",  # type: ignore
    ):
        self.quantile_range = quantile_range
        self.nan_policy = nan_policy

    def fit(self, X, y=None) -> "Winsorizer":

        if hasattr(X, "columns"):
            self.feature_names_in_ = list(X.columns)
            self.feature_names_out_ = self.feature_names_in_

        # Check X,y shape
        X = check_array(X, accept_sparse=True)
        self.n_features_in_ = X.shape[1]

        # Compute quantiles
        self.quantiles_ = np.nanquantile(X, self.quantile_range, axis=0, interpolation="linear")
        self.n_features_out_ = self.n_features_in_

        return self

    @staticmethod
    def _winsorize_1d(v: np.ndarray, low_value: np.number, up_value: np.number) -> np.ndarray:
        v[v <= low_value] = low_value
        v[v >= up_value] = up_value
        return v

    def transform(self, X) -> np.ndarray:

        check_is_fitted(self, "quantiles_")

        X = check_array(X, accept_sparse=True)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                "Unexpected input shape. Got %d, expected %d (from fit)" % (X.shape[1], self.n_features_in_)
            )

        for i in range(self.n_features_in_):
            self._winsorize_1d(X[:, i], self.quantiles_[0, i], self.quantiles_[1, i])
        return X

    def _more_tags(self) -> Dict[str, Any]:
        return {"allow_nan": True, "X_types": ["2darray", "2dlabels"]}
