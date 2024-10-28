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


class PRAG(Metric):
    def __init__(self) -> None:
        """
        This class computes Pairwise Ranking Accuracy Gap (PRAG).
        """
        self.name = "PRAG"

    def evaluate(self, list1: List[str], list2: List[str]) -> float:
        """
        This method computes PRAG score between two lists.

        Parameters
        ----------
        list1 : list of strings
            A list of recommendation from an LLM model.

        list2 : list of strings
            Another list of recommendation from an LLM model.

        Returns
        -------
        float
            PRAG metric value for the two provided lists of recommendations (float)
        """
        K = len(list1)
        if not list1 or not list2:
            return 0

        if len(list1) == 1:
            return int(list1 == list2)

        matches = 0
        for i, item1 in enumerate(list1):
            for j, item2 in enumerate(list1[i + 1 :], start=i + 1):
                id1, id2 = (
                    list2.index(item) if item in list2 else -1
                    for item in (item1, item2)
                )
                if id1 != -1 and (id1 < id2 or id2 == -1):
                    matches += 1

        return matches / (K * (K + 1))
