<p align="center">
  <img src="https://raw.githubusercontent.com/cvs-health/langfair/main/assets/images/langfair-logo.png" />
</p>

# Use-Case Level LLM Bias and Fairness Assessments

LangFair is a Python library for conducting bias and fairness assessments of large language model (LLM) use cases. This repository includes a framework for [choosing bias and fairness metrics](https://github.com/cvs-health/langfair/tree/main#choosing-bias-and-fairness-metrics-for-an-llm-use-case), [demo notebooks](https://github.com/cvs-health/langfair/tree/main/examples), and a LLM bias and fairness [technical playbook](https://arxiv.org/abs/2407.10853) containing a thorough discussion of LLM bias and fairness risks, evaluation metrics, and best practices. Please refer to our [documentation site](https://cvs-health.github.io/langfair/) for more details on how to use LangFair.

#### Why LangFair?
Static benchmark assessments, which are typically assumed to be sufficiently representative, often fall short in capturing the risks associated with all possible use cases of LLMs. These models are increasingly used in various applications, including recommendation systems, classification, text generation, and summarization. However, evaluating these models without considering use-case-specific prompts can lead to misleading assessments of their performance, especially regarding bias and fairness risks.
 
LangFair addresses this gap by adopting a Bring Your Own Prompts (BYOP) approach, allowing users to tailor bias and fairness evaluations to their specific use cases. This ensures that the metrics computed reflect the true performance of the LLMs in real-world scenarios, where prompt-specific risks are critical. Additionally, LangFair's focus is on output-based metrics that are practical for governance audits and real-world testing, without needing access to internal model states.

#### Supported Bias and Fairness Metrics
Bias and fairness metrics offered by LangFair are grouped into several categories. The full suite of metrics is displayed below.

##### Toxicity Metrics
* Expected Maximum Toxicity ([Gehman et al., 2020](https://arxiv.org/abs/2009.11462))
* Toxicity Probability ([Gehman et al., 2020](https://arxiv.org/abs/2009.11462))
* Toxic Fraction ([Liang et al., 2023](https://arxiv.org/abs/2211.09110))

##### Counterfactual Fairness Metrics
* Strict Counterfactual Sentiment Parity ([Huang et al., 2020](https://arxiv.org/abs/1911.03064))
* Weak Counterfactual Sentiment Parity ([Bouchard, 2024](https://arxiv.org/abs/2407.10853))
* Counterfactual Cosine Similarity Score ([Bouchard, 2024](https://arxiv.org/abs/2407.10853))
* Counterfactual BLEU ([Bouchard, 2024](https://arxiv.org/abs/2407.10853))
* Counterfactual ROUGE-L ([Bouchard, 2024](https://arxiv.org/abs/2407.10853))

##### Stereotype Metrics
* Stereotypical Associations ([Liang et al., 2023](https://arxiv.org/abs/2211.09110))
* Co-occurrence Bias Score ([Bordia & Bowman, 2019](https://arxiv.org/abs/1904.03035))
* Stereotype classifier metrics ([Zekun et al., 2023](https://arxiv.org/abs/2311.14126), [Bouchard, 2024](https://arxiv.org/abs/2407.10853))

##### Recommendation (Counterfactual) Fairness Metrics
* Jaccard Similarity ([Zhang et al., 2023](https://dl.acm.org/doi/10.1145/3604915.3608860))
* Search Result Page Misinformation Score ([Zhang et al., 2023](https://dl.acm.org/doi/10.1145/3604915.3608860))
* Pairwise Ranking Accuracy Gap ([Zhang et al., 2023](https://dl.acm.org/doi/10.1145/3604915.3608860))

##### Classification Fairness Metrics
* Predicted Prevalence Rate Disparity ([Feldman et al., 2015](https://arxiv.org/abs/1412.3756); [Bellamy et al., 2018](https://arxiv.org/abs/1810.01943); [Saleiro et al., 2019](https://arxiv.org/abs/1811.05577))
* False Negative Rate Disparity ([Bellamy et al., 2018](https://arxiv.org/abs/1810.01943); [Saleiro et al., 2019](https://arxiv.org/abs/1811.05577))
* False Omission Rate Disparity ([Bellamy et al., 2018](https://arxiv.org/abs/1810.01943); [Saleiro et al., 2019](https://arxiv.org/abs/1811.05577))
* False Positive Rate Disparity ([Bellamy et al., 2018](https://arxiv.org/abs/1810.01943); [Saleiro et al., 2019](https://arxiv.org/abs/1811.05577))
* False Discovery Rate Disparity ([Bellamy et al., 2018](https://arxiv.org/abs/1810.01943); [Saleiro et al., 2019](https://arxiv.org/abs/1811.05577))




## Quickstart 
### (Optional) Create a virtual environment for using LangFair
We recommend creating a new virtual environment using venv before installing LangFair. To do so, please follow instructions [here](https://docs.python.org/3/library/venv.html).

### Installing LangFair
The latest version can be installed from PyPI:

```bash
pip install langfair
```

### Usage
Below is a sample of code illustrating how to use LangFair's `AutoEval` class for text generation and summarization use cases. The below example assumes the user has already defined parameters `DEPLOYMENT_NAME`, `API_KEY`, `API_BASE`, `API_TYPE`, `API_VERSION`, and a list of prompts from their use case `prompts`.

Create `langchain` LLM object.
```python
from langchain_openai import AzureChatOpenAI
llm = AzureChatOpenAI(
    deployment_name=DEPLOYMENT_NAME,
    openai_api_key=API_KEY,
    azure_endpoint=API_BASE,
    openai_api_type=API_TYPE,
    openai_api_version=API_VERSION,
    temperature=0.4 # User to set temperature
)
```

Run the `AutoEval` method for automated bias / fairness evaluation
```python
from langfair.auto import AutoEval
auto_object = AutoEval(
    prompts=prompts, 
    langchain_llm=llm
    # toxicity_device=device # use if GPU is available
)

results = await auto_object.evaluate()
```

<p align="center">
  <img src="https://raw.githubusercontent.com/cvs-health/langfair/develop/assets/images/autoeval_process.png" />
</p>


Print the results and export to .txt file.
```python
auto_object.export_results(file_name="metric_values.txt")
auto_object.print_results()
```

<p align="center">
  <img src="https://raw.githubusercontent.com/cvs-health/langfair/main/assets/images/autoeval_output.png" />
</p>

## Example Notebooks
See **[Demo Notebooks](https://github.com/cvs-health/langfair/tree/main/examples)** for notebooks illustrating how to use LangFair for various bias and fairness evaluation metrics.

## Choosing Bias and Fairness Metrics for an LLM Use Case
In general, bias and fairness assessments of LLM use cases do not require satisfying all possible evaluation metrics. Instead, practitioners should prioritize and focus on a relevant subset of metrics. To demystify metric choice for these assessments, a decision framework for selecting the appropriate evaluation metrics is depicted in the diagram below. For more details on this decision framework, we refer users to our [technical playbook](https://arxiv.org/abs/2407.10853).

The scope of this framework is restricted to use cases for which prompts can be sampled from a known population
and the task is well-defined. Use cases are categorized into three distinct groups based on task: 1) text generation
and summarization, 2) classification, and 3) recommendation. Within each category, metrics to assess pertinent bias and fairness risks are identified based on characteristics of the use case. The specifics of this mapping are elaborated upon below.

When considering text generation and summarization, an important factor in determining relevant bias and fairness metrics is whether the use case upholds fairness through unawareness (FTU). This means that all inputs provided to the model do not include any mentions of protected attribute words. If FTU cannot be achieved, it is recommended that practitioners include counterfactual fairness and stereotype metrics in their assessments. Additionally, it is recommended that all text generation and summarization use cases undergo toxicity evaluation, regardless of whether or not FTU is achieved. For illustrations of how to calculate these metrics with LangFair, refer to the [toxicity, stereotype, and counterfactual demo notebooks](https://github.com/cvs-health/langfair/tree/main/examples/evaluations/text_generation) or the [AutoEval demo notebook](https://github.com/cvs-health/langfair/tree/main/examples/evaluations/text_generation/auto_eval_demo.ipynb) for a more automated implementation.

To determine suitable bias and fairness metrics for text classification use cases, a modified version of the *Aequitas* decision framework proposed by [Saleiro et al., 2018](https://arxiv.org/abs/1811.05577) is adopted. This framework can be applied to any classification use case where inputs correspond to protected attribute groups. In such cases, the following approach is recommended: if fairness necessitates that model predictions exhibit approximately equal predicted prevalence across different groups, representation fairness metrics should be used; Otherwise, error-based fairness metrics should be used. Among error-based metrics, practitioners should focus on disparities in false negatives (measured by false negative rate disparity and false omission rate disparity) for assistive use cases (where interventions help individuals). Conversely, for punitive use cases (where interventions can harm individuals), practitioners should focus on disparities in false positives (measured by false positive rate disparity and false discovery rate disparity). If inputs cannot be mapped to a protected attribute, meaning they are not person-level inputs and they satisfy FTU, a fairness assessment is not applicable.  For a simple illustration of how to calculate classification fairness metrics with LangFair, users may refer to the [classification metrics demo notebook](https://github.com/cvs-health/langfair/tree/main/examples/evaluations/classification/classification_metrics_demo.ipynb).

Lastly, for recommendation use cases, counterfactual unfairness is a risk if FTU cannot be satisfied, as shown by [Zhang et al., 2023](https://dl.acm.org/doi/10.1145/3604915.3608860). For these scenarios, it is recommended practitioners assess counterfactual fairness in recommendations using the recommendation fairness metrics proposed by [Zhang et al., 2023](https://dl.acm.org/doi/10.1145/3604915.3608860). However, if FTU is satisfied, then a fairness assessment is not applicable. For illustrations of how to compute these metrics with LangFair, refer to the [recommendation metrics demo notebook](https://github.com/cvs-health/langfair/tree/main/examples/evaluations/recommendation/recommendation_metrics_demo.ipynb).


<p align="center">
  <img src="https://raw.githubusercontent.com/cvs-health/langfair/main/assets/images/use_case_framework.PNG" />
</p>


## Associated Research
A technical description of LangFair's evaluation metrics and a practitioner's guide for selecting evaluation metrics is contained in **[this paper](https://arxiv.org/abs/2407.10853)**. Below is the bibtex entry for this paper:

```bibtex
@misc{bouchard2024actionableframeworkassessingbias,
      title={An Actionable Framework for Assessing Bias and Fairness in Large Language Model Use Cases}, 
      author={Dylan Bouchard},
      year={2024},
      eprint={2407.10853},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2407.10853}, 
}
```

## Code Documentation
Please refer to our [documentation site](https://cvs-health.github.io/langfair/) for more details on how to use LangFair.

## Development Team
The open-source version of LangFair is the culmination of extensive work carried out by a dedicated team of developers. While the internal commit history will not be made public, we believe it's essential to acknowledge the significant contributions of our development team who were instrumental in bringing this project to fruition:

- [Dylan Bouchard](https://github.com/dylanbouchard)
- [Mohit Singh Chauhan](https://github.com/mohitcek)
- [David Skarbrevik](https://github.com/dskarbrevik)
- [Viren Bajaj](https://github.com/virenbajaj)
- [Zeya Ahmad](https://github.com/zeya30)

## Contributing
Contributions are welcome. Please refer [here](https://github.com/cvs-health/langfair/tree/main/CONTRIBUTING.md) for instructions on how to contribute to LangFair.