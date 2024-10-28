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

from langfair.metrics.recommendation.metrics.baseclass.metrics import Metric


class JaccardSimilarity(Metric):
    def __init__(self) -> None:
        """
        This class computes the Jaccard Similarity score.
        """
        self.name = "Jaccard"

    def evaluate(self, list1: List[str], list2: List[str]) -> float:
        """
        This method computes Jaccard Similarity between two lists.

        Parameters
        ----------
        list1 : list of strings
            A list of recommendation from an LLM model.

        list2 : list of strings
            Another list of recommendation from an LLM model.

        Returns
        -------
        float
            Jaccard similarity for the two provided lists of recommendations (float)
        """
        x = set(list1)
        y = set(list2)
        return len(x & y) / len(x | y)
