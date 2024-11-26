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

from typing import Dict, Optional

from numpy.typing import ArrayLike

from langfair.metrics.classification import metrics

AssistiveMetricObjects = {
    "FNRP": metrics.FalseNegativeRateParity(),
    "FORP": metrics.FalseOmissionRateParity(),
}
PunitiveMetricObjects = {
    "FPRP": metrics.FalsePositiveRateParity(),
    "FDRP": metrics.FalseDiscoveryRateParity(),
}
RepresenationMetricObjects = {
    "PPRP": metrics.PredictedPrevalenceRateParity(),
}
MetricTypeToDict = {
    "assistive": AssistiveMetricObjects,
    "punitive": PunitiveMetricObjects,
    "representation": RepresenationMetricObjects,
    "all": {
        **AssistiveMetricObjects,
        **PunitiveMetricObjects,
        **RepresenationMetricObjects,
    },
}


class ClassificationMetrics:
    def __init__(self, metric_type: str = "all") -> None:
        """
        Class for pairwise classification fairness metrics. For more information on these metrics,
        see Feldman et al. (2015) :footcite:`feldman2015certifyingremovingdisparateimpact`,
        Bellamy et al. (2018) :footcite:`bellamy2018aifairness360extensible` and Saleiro et al. (2019) :footcite:`saleiro2019aequitasbiasfairnessaudit`.

        Parameters
        ----------
        metric_type: str, one of 'assistive', 'punitive', 'representation', or 'all', default='all'
            A list containing name or class object of metrics.
        """
        assert metric_type in ["all", "assistive", "punitive", "representation"], """
        metric type must be one of 'all', 'assistive', 'punitive', 'representation'
        """
        self.metric_type = metric_type
        self._create_instances()

    def evaluate(
        self,
        groups: ArrayLike,
        y_pred: ArrayLike,
        y_true: Optional[ArrayLike] = None,
        ratio: bool = False,
    ) -> Dict[str, float]:
        """
        Returns values of classification fairness metrics

        Parameters
        ----------
        groups : Array-like
            Group indicators. Must contain exactly two unique values.

        y_pred : Array-like
            Binary model predictions. Positive and negative predictions must be 1 and 0, respectively.

        y_true : Array-like, default=None
            Binary labels (ground truth values). Positive and negative labels must be 1 and 0, respectively.

        ratio : bool, default=False
            Indicates whether to compute the metric as a difference or a ratio

        Returns
        -------
        dict
            Dictionary containing specified metric values

        References
        ----------
        .. footbibliography::
        """
        return {
            metric.name: metric.evaluate(
                groups=groups, y_pred=y_pred, y_true=y_true, ratio=ratio
            )
            for metric in self.metrics
        }

    def _create_instances(self) -> None:
        """Define default metrics."""
        metric_dict = MetricTypeToDict[self.metric_type]
        self.metrics = []
        for name in metric_dict:
            self.metrics.append(metric_dict[name])
