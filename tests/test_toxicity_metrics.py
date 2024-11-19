import json
import unittest
from math import isclose

import torch

from langfair.metrics.toxicity import AvailableClassifiers, ToxicityMetrics


class TestToxicityMetrics(unittest.TestCase):
    def test_evaluate(self):
        toxic_responses_file = "toxic_responses.csv"
        data_folder = "tests/data/toxicity"  # TODO: convert to Path
        with open(f"{data_folder}/{toxic_responses_file}", "r") as f:
            toxic = json.load(f)
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )  # Use GPU if available
        for classifier in AvailableClassifiers:
            print(f"Classifier:{classifier}")
            detoxify = ToxicityMetrics(
                classifiers=[classifier],
                batch_size=100,
                device=device,
                toxic_threshold=0.01,
            )  # TODO: download and pass the model
            toxic_results = detoxify.evaluate(
                responses=toxic.get("data").get("response"),
                prompts=toxic.get("data").get("prompt"),
            )  # TODO: make print statements optional while  computing scores
            print(f"Calculated result:{toxic_results}")
            with open(f"{data_folder}/toxic_results_{classifier}.json", "r") as f:
                toxic_results_actual = json.load(f)
                print(f"Actual result:{toxic_results_actual}")

            assert isclose(
                toxic_results.get("metrics").get("Toxic Fraction"),
                toxic_results_actual.get("metrics").get("Toxic Fraction"),
            )

            assert isclose(
                toxic_results.get("metrics").get("Expected Maximum Toxicity"),
                toxic_results_actual.get("metrics").get("Expected Maximum Toxicity"),
                abs_tol=1e-06,
            )  # Tolerance ensures equality #TODO: figure out why the same call from different files produces different results

            assert isclose(
                toxic_results.get("metrics").get("Toxicity Probability"),
                toxic_results_actual.get("metrics").get("Toxicity Probability"),
            )


if __name__ == "__main__":
    unittest.main()
