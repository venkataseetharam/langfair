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

from typing import List

from langfair.metrics.utils.classifier_metrics.baseclass.metrics import Metric


class Fraction(Metric):
    def __init__(self):
        """
        This class computes the Fraction metric for toxicity and stereotype assessment.
        """
        super().__init__()
        self.name = "fraction"

    @staticmethod
    def metric_function(scores: List[float], toxic_threshold: float) -> float:
        """
        Calculate fraction metric from a list of scores.

        Parameters
        ----------
        scores: list of floats
            A dictionary containing toxicity/stereotype score.

        toxic_threshold : float
            Specifies the threshold to use for toxicity/stereotype score classification.

        Returns
        -------
        float
            Value of toxicity/stereotype fraction
        """
        toxic_scores = [score for score in scores if score > toxic_threshold]
        return len(toxic_scores) / len(scores)
