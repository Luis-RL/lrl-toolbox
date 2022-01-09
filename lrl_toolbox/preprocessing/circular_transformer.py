from typing import Any, Dict, List, Optional

import numpy as np
from pydantic.typing import Literal
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted


class CricularTransformer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        period: Optional[List[np.number]] = None,
        nan_policy: Literal["propagate", "raise", "omit"] = "propagate",  # type: ignore
    ):
        self.period = period
        self.nan_policy = nan_policy

    def fit(self, X, y=None) -> "CricularTransformer":

        if hasattr(X, "columns"):
            self.feature_names_in_ = list(X.columns)
            cos_names = ["cos_" + f for f in self.feature_names_in_]
            sin_names = ["sin_" + f for f in self.feature_names_in_]
            self.feature_names_out_ = cos_names + sin_names

        # Check X,y shape
        X = check_array(X, accept_sparse=True)
        self.n_features_in_ = X.shape[1]
        self.n_features_out_ = self.n_features_in_ * 2

        # Compute period
        if self.period is None:
            self.period_ = np.max(X, axis=0) - np.min(X, axis=0)
        else:
            if len(self.period) != self.n_features_in_:
                raise ValueError(
                    "Unexpected input shape. Got %d, expected %d (period arg)" % (X.shape[1], self.n_features_in_)
                )
            self.period_ = np.array(self.period)

        return self

    def transform(self, X) -> np.ndarray:

        check_is_fitted(self, "period_")

        X = check_array(X, accept_sparse=True)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                "Unexpected input shape. Got %d, expected %d (from fit)" % (X.shape[1], self.n_features_in_)
            )

        X = X / self.period_ * (2 * np.pi)
        X_cos = np.cos(X)
        X_sin = np.sin(X)
        return np.concatenate([X_cos, X_sin], axis=1)

    def _more_tags(self) -> Dict[str, Any]:
        return {"allow_nan": True, "X_types": ["2darray", "2dlabels"]}
