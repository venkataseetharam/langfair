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

from abc import ABC, abstractmethod
from typing import List


class Metric(ABC):
    """
    Abstract base class of all recommendation metrics. Serves as a template for creating new metric
    functions.
    """

    @abstractmethod
    def evaluate(self, list1: List[str], list2: List[str]) -> float:
        """
        Abstract method that needs to be implemented by the user when creating a new metric function.
        """
        pass
