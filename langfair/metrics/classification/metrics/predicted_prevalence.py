# Copyright 2024 CVS Health and/or one of its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional

import numpy as np
from numpy.typing import ArrayLike

from langfair.metrics.classification.metrics.baseclass.metrics import Metric


class PredictedPrevalenceRateParity(Metric):
    def __init__(self) -> None:
        """
        This class computes predicted prevalence rate parity. The user may specify whether to compute this
        metric as a difference or a ratio. For more information on these metrics, see Feldman et al. (2015) :footcite:`feldman2015certifyingremovingdisparateimpact`.
        """
        self.name = "PredictedPrevalenceRateParity"

    def evaluate(
        self,
        groups: ArrayLike,
        y_pred: ArrayLike,
        y_true: Optional[ArrayLike] = None,
        ratio: bool = False,
    ) -> float:
        """
        This method computes disparity in predicted prevalence rates between two groups.

        Parameters
        ----------
        groups : Array-like
            Group indicators. Must contain exactly two unique values.

        y_pred : Array-like
            Binary model predictions. Positive and negative predictions must be 1 and 0, respectively.

        y_true : Array-like, default=None
            Binary labels (ground truth values). This argument is only a placeholder for this class.

        ratio : bool, default=False
            Indicates whether to compute the metric as a difference or a ratio

        Returns
        -------
        float
            Value of predicted prevalence rate parity

        References
        ----------
        .. footbibliography::
        """
        unique_preds = np.unique(y_pred)
        assert np.array_equal(
            unique_preds, [0, 1]
        ), "y_pred must contain exactly two unique values: 0 and 1"

        unique_groups = np.unique(groups)
        assert len(unique_groups) == 2, "groups must contain exactly two unique values"

        ppr1 = np.mean(y_pred[groups == unique_groups[0]])
        ppr2 = np.mean(y_pred[groups == unique_groups[1]])

        return ppr1 / ppr2 if ratio else abs(ppr1 - ppr2)
