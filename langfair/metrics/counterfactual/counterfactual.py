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

from typing import Union

from langfair.generator.counterfactual import CounterfactualGenerator
from langfair.metrics.counterfactual import metrics
from langfair.metrics.counterfactual.metrics.baseclass.metrics import Metric

MetricType = Union[list[str], list[Metric]]
DefaultMetricObjects = {
    "Cosine": metrics.CosineSimilarity(transformer="all-MiniLM-L6-v2"),
    "Rougel": metrics.RougelSimilarity(),
    "Bleu": metrics.BleuSimilarity(),
    "Sentiment Bias": metrics.SentimentBias(),
}
DefaultMetricNames = list(DefaultMetricObjects.keys())


################################################################################
# Calculate Counterfactual Metrics
################################################################################
class CounterfactualMetrics:
    def __init__(
        self, metrics: MetricType = DefaultMetricNames, neutralize_tokens: str = True
    ) -> None:
        """
        This class computes few or all counterfactual metrics supported LangFair.

        Parameters
        ----------
        metrics: list of string/objects, default=["Cosine", "Rougel", "Bleu", "Sentiment Bias"]
            A list containing name or class object of metrics.

        neutralize_tokens: boolean, default=True
            An indicator attribute to use masking for the computation of Blue and RougeL metrics. If True, counterfactual
            responses are masked using `CounterfactualGenerator.neutralize_tokens` method before computing the aforementioned metrics.
        """
        self.metrics = metrics
        if isinstance(metrics[0], str):
            self.metric_names = metrics
            self._validate_metrics(metrics)
            self._default_instances()
        self.neutralize_tokens = neutralize_tokens
        if self.neutralize_tokens:
            self.cf_generator = CounterfactualGenerator()

    def evaluate(self, texts1: list, texts2: list, attribute: str = None):
        """
        This method evaluate the counterfactual metrics values for the provided pair of texts.

        Parameters
        ----------
        texts1 : list of strings
            A list of generated outputs from a language model each containing mention of the
            same protected attribute group.

        texts2 : list of strings
            A list, analogous to `texts1` of counterfactually generated outputs from a language model each containing
            mention of the same protected attribute group. The mentioned protected attribute must be a different group
            within the same protected attribute as mentioned in `texts1`.

        attribute : {'gender', 'race'}, default='gender'
            Specifies whether to use race or gender for neutralization

        Returns
        ----------
        Dictionary containing values of counterfactual metrics
        """
        if self.neutralize_tokens:
            assert attribute in [
                "gender",
                "race",
            ], """langfair: To neutralize tokens, 'attribute' should 
            be either "gender" or "race"."""
            masked_texts1 = self.cf_generator.neutralize_tokens(
                texts=texts1, attribute=attribute
            )
            masked_texts2 = self.cf_generator.neutralize_tokens(
                texts=texts2, attribute=attribute
            )
        metric_values = {}
        for metric in self.metrics:
            if (
                metric.name in ["Bleu Similarity", "RougeL Similarity"]
                and self.neutralize_tokens
            ):
                metric_values[metric.name] = metric.evaluate(
                    texts1=masked_texts1, texts2=masked_texts2
                )
            else:
                metric_values[metric.name] = metric.evaluate(
                    texts1=texts1, texts2=texts2
                )
        return metric_values

    def _default_instances(self):
        """Define default metrics."""
        self.metrics = []
        for name in self.metric_names:
            self.metrics.append(DefaultMetricObjects[name])

    def _validate_metrics(self, metric_names):
        """Validate that specified metrics metrics are supported."""
        for name in metric_names:
            assert (
                name in DefaultMetricNames
            ), """langfair: Provided metric name is not part of available metrics."""
