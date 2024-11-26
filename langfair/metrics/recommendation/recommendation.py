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

from typing import Any, Dict, List, Union

import numpy as np

from langfair.metrics.recommendation import metrics
from langfair.metrics.recommendation.metrics.baseclass.metrics import Metric

MetricType = Union[list[str], list[Metric]]
DefaultMetricObjects = {
    "Jaccard": metrics.JaccardSimilarity(),
    "PRAG": metrics.PRAG(),
    "SERP": metrics.SERP(),
}
DefaultMetricNames = list(DefaultMetricObjects.keys())


################################################################################
# Class to evaluate FaiRLLM Metrics
################################################################################
class RecommendationMetrics:
    def __init__(self, metrics: MetricType = DefaultMetricNames) -> None:
        """
        Class for LLM recommendation fairness metrics. Compute FaiRLLM (Fairness of Recommendation via LLM) metrics. This class
        enables calculation of Jaccard Similarity, SEarch Result Page Misinformation Score (SERP),
        and Pairwise Ranking Accuracy Gap (PRAG) across protected attribute groups.

        For more information on these metrics, refer to Zhang et al. (2023): :footcite:`Zhang_2023, series={RecSys ’23}`.

        Parameters
        ----------
        metrics: list of string/objects, default=["Jaccard", "PRAG", "SERP"]
            A list containing name or class object of metrics.
        """

        self.metrics = metrics
        if isinstance(metrics[0], str):
            self.metric_names = metrics
            self._validate_metrics(metrics)
            self._default_instances()

    def evaluate_against_neutral(
        self,
        group_dict_list: List[Dict[str, List[str]]],
        neutral_dict: Dict[str, List[str]],
    ) -> Dict[str, Any]:
        """
        Returns min, max, range, and standard deviation of SERP, Jaccard,
        and PRAG similarity metrics across protected attribute groups. Metrics
        are consistent with those provided by https://arxiv.org/pdf/2305.07609.pdf

        Parameters
        ----------
        neutral_dict : dictionary of lists
            Each value in the list corresponds to a recommendation list. For example,
            neutral_dict = {
                 'TS': [
                    "Love Story",
                    "You Belong with Me",
                    "Blank Space",
                    "Shake It Off",
                    "Bad Blood",
                    "Style",
                    "Wildest Dreams",
                    "Delicate",
                    "ME!",
                    "Cardigan"
                ],
                'ES': [
                    "The A Team",
                    "Thinking Out Loud",
                    "Shape of You",
                    "Castle on the Hill",
                    "Perfect",
                    "Photograph",
                    "Dive",
                    "Galway Girl",
                    "Happier",
                    "Lego House"
                ]
            }

        group_dict_list : list of dictionaries of lists
            Each element of the list corresponds to a protected attribute group. The values of each
            interior dictionary are recommendation lists in the format of `neutral_dict`. For example,
            group_dict_list = [
                {
                    'TS': [
                        "Love Story",
                        "Shake It Off",
                        "Blank Space",
                        "You Belong with Me",
                        "Bad Blood",
                        "Style",
                        "Wildest Dreams",
                        "Delicate",
                        "Look What You Made Me Do",
                        "We Are Never Ever Getting Back Together"
                    ],
                    'ES': [
                        "The A Team",
                        "Thinking Out Loud",
                        "Shape of You",
                        "Castle on the Hill",
                        "Perfect",
                        "Photograph",
                        "Dive",
                        "Sing",
                        "Galway Girl",
                        "I Don't Care (with Justin Bieber)"
                    ]
                },
                {
                    'TS': [
                        "Love Story",
                        "You Belong with Me",
                        "Blank Space",
                        "Shake It Off",
                        "Style",
                        "Wildest Dreams",
                        "Delicate",
                        "ME!",
                        "Cardigan",
                        "Folklore",
                        ],
                    'ES': [
                        "Castle on the Hill",
                        "Perfect",
                        "Shape of You",
                        "Thinking Out Loud",
                        "Photograph",
                        "Galway Girl",
                        "Dive",
                        "Happier",
                        "Lego House",
                        "Give Me Love"
                    ]
                }
            ]

        Returns
        -------
        dict
            Dictionary containing mean, max, standard deviation, and range for
            Jaccard, SERP, PRAG across protected attribute groups
        """
        self._run_input_checks(group_dict_list, neutral_dict)
        return self._return_min_max_delta_std(
            neutral_dict=neutral_dict, group_dict_list=group_dict_list
        )

    def evaluate_pairwise(
        self, rec_lists1: List[List[str]], rec_lists2: List[List[str]]
    ) -> Dict[str, float]:
        """
        Returns pairwise values of SERP, Jaccard, and PRAG similarity metrics
        for two protected attribute groups. Metrics are pairwise analogs of
        those provided by https://arxiv.org/pdf/2305.07609.pdf

        Parameters
        ----------
        rec_lists1 : list of lists of strings
            A list of recommendation lists, each of length K, generated from prompts containing
            mentions of the same protected attribute group.

        rec_lists2 : list of lists of strings
            A list of recommendation lists, each of length K, generated from prompts containing
            mentions of the same protected attribute group. Prompts should be identical to those
            used to generate `rec_lists1` except they should mention a different protected
            attribute group.

        Returns
        -------
        dict
            Dictionary containing pairwise metric values of SERP, Jaccard, and PRAG

        References
        ----------
        .. footbibliography::
        """
        assert len(rec_lists1) == len(
            rec_lists2
        ), "The same number of recommendation lists must be provided for both groups"
        K = len(rec_lists1[0])
        for i in range(len(rec_lists1)):
            assert (
                len(rec_lists1[i]) == K
            ), "Recommendation lists must all be of equal length"
            assert (
                len(rec_lists2[i]) == K
            ), "Recommendation lists must all be of equal length"
        return {
            metric.name: self._pairwise_calculations(rec_lists1, rec_lists2, metric)
            for metric in self.metrics
        }

    def _return_min_max_delta_std(
        self,
        group_dict_list: List[Dict[str, List[str]]],
        neutral_dict: Dict[str, List[str]],
    ) -> Dict[str, float]:
        """
        Helper function to evaluate min, max, min-max delta, and stardard deviation
        of similarity metric values across protected attribute groups
        """
        # Initialize lists and results dictionary
        result_dict = {}

        # Evaluate each similarity metric for each group
        for metric in self.metrics:
            group_metric_value_list = []
            for group_dict in group_dict_list:
                group_metric_value_list.append(
                    self._get_metric_with_neutral(
                        group_dict=group_dict,
                        neutral_dict=neutral_dict,
                        metric=metric,
                    )
                )
            group_metric_value_list = np.array(group_metric_value_list)
            name = metric.name
            # Save max, min, delta, std for current metric
            result_dict[name] = {}
            result_dict[name]["max"] = group_metric_value_list.max()
            result_dict[name]["min"] = group_metric_value_list.min()
            result_dict[name]["SNSR"] = (
                group_metric_value_list.max() - group_metric_value_list.min()
            )
            result_dict[name]["SNSV"] = group_metric_value_list.std()

        return result_dict

    def _default_instances(self) -> None:
        """Define default metrics."""
        self.metrics = []
        for name in self.metric_names:
            self.metrics.append(DefaultMetricObjects[name])

    def _validate_metrics(self, metric_names: List[str]) -> None:
        """Validate that specified metrics are supported."""
        for name in metric_names:
            assert (
                name in DefaultMetricNames
            ), """Provided metric name is not part of available metrics."""

    @staticmethod
    def _run_input_checks(
        group_dict_list: List[Dict[str, List[str]]],
        neutral_dict: Dict[str, List[str]] = None,
    ) -> None:
        """
        Helper function to run input checks.
        """
        # High level type check
        type_msg = "Please ensure input types are correct. For assistance, run `help(RecommendationMetric)`."
        assert isinstance(group_dict_list, list), type_msg
        K = (
            len(neutral_dict[list(neutral_dict.keys())[0]])
            if neutral_dict is not None
            else len(group_dict_list[0].keys())
        )

        # Assert recommendation lists are of equal length
        if neutral_dict is not None:
            assert isinstance(neutral_dict, dict), type_msg
            for key in neutral_dict.keys():
                assert isinstance(neutral_dict[key], list), type_msg
                assert (
                    len(neutral_dict[key]) == K
                ), "Recommendation lists must all be of equal length."
        for group_dict in group_dict_list:
            assert isinstance(group_dict, dict), type_msg
            for key in group_dict.keys():
                assert isinstance(group_dict[key], list), type_msg
                assert (
                    len(group_dict[key]) == K
                ), "Recommendation lists must all be of equal length."
        return

    @staticmethod
    def _get_metric_with_neutral(
        group_dict: Dict[str, List[str]],
        neutral_dict: Dict[str, List[str]],
        metric: Any,
    ) -> float:
        """
        Helper function to calculate SERP similarity metric
        """
        metric_value_list = []
        for rec_list in group_dict.keys():
            compare_list = group_dict[rec_list]
            neutral_list = neutral_dict[rec_list]
            metric_value_list.append(
                metric.evaluate(list1=compare_list, list2=neutral_list)
            )
        return np.mean(metric_value_list)

    @staticmethod
    def _pairwise_calculations(
        rec_lists1: List[str], rec_lists2: List[str], metric: Any
    ) -> float:
        """
        Helper function to calculate SERP similarity metric (pairwise)
        """
        val_list = []
        for i in range(len(rec_lists1)):
            val_i = min(
                metric.evaluate(rec_lists1[i], rec_lists2[i]),
                metric.evaluate(rec_lists1[i], rec_lists2[i]),
            )
            val_list.append(val_i)
        return np.mean(val_list)
