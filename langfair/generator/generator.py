# Copyright 2024 CVS Health and/or one of its affiliates
#
# Copyright 2023 OpenAI
#
# Licensed under the MIT License.
#
# The original work of OpenAI has been modified
# by CVS Health to include functionality for computing
# prompt and response token counts for OpenAI models.

import asyncio
import itertools
import random
import time
import warnings
from typing import Any, Dict, List, Optional, Tuple

import langchain_core
import langchain_openai
import numpy as np
import openai
import tiktoken
from langchain.chains import LLMChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)

from langfair.constants.cost_data import COST_MAPPING

FAILURE_MESSAGE = "Unable to get response"
TOKEN_COST_DATE = "08/20/2024"
warnings.filterwarnings("ignore", category=DeprecationWarning)


class ResponseGenerator:
    def __init__(
        self,
        langchain_llm: Any = None,
        max_calls_per_min: Optional[int] = None,
    ) -> None:
        """
        Class for generating data from a provided set of prompts

        Parameters
        ----------
        langchain_llm : langchain llm object, default=None
            A langchain llm object to get passed to chain constructor. User is responsible for specifying
            temperature and other relevant parameters to the constructor of their `langchain_llm` object.

        max_calls_per_min : int, default=None
            Specifies how many api calls to make per minute to avoid a rate limit error. By default, no
            limit is specified.

        Attributes
        ----------
        cost_mapping : dict
            A dictionary containing the cost information for different models. The keys are the model names, and the values are
            dictionaries specifying the input and output costs per token.
        """
        self.llm = langchain_llm
        self.cost_mapping = COST_MAPPING
        self.max_calls_per_min = max_calls_per_min
        self.failure_message = FAILURE_MESSAGE
        self.token_cost_date = TOKEN_COST_DATE

    async def estimate_token_cost(
        self,
        tiktoken_model_name: str,
        prompts: List[str],
        example_responses: List[str] = None,
        response_sample_size: int = 30,
        system_prompt: str = "You are a helpful assistant",
        count: int = 25,
    ) -> Dict[str, float]:
        """
        Estimates the token cost for a given list of prompts and (optionally) example responses.
        Note: This method is only compatible with GPT models. Cost-per-token values are as of
        08/07/2024.

        Parameters
        ----------
        tiktoken_model_name: str
           The name of the OpenAI model to use for token counting.

        prompts : list of strings
           A list of prompts

        example_responses : list of strings, default=None
           A list of example responses. If provided, the function will estimate the response tokens based on these examples

        response_sample_size : int, default=30.
           The number of responses to generate for cost estimation if `example_responses` is not provided.

        system_prompt : str, default="You are a helpful assistant."
           Specifies the system prompt used when generating LLM responses.

        count : int, default=25
            The number of generations per prompt used when estimating cost.

        Returns
        -------
        dict
           A dictionary containing the estimated token costs, including prompt token cost, completion token cost,
           and total token cost.
        """
        # TODO: Add token costs for other models
        # TODO: Scrape rather than hard-code costs.
        print(
            f"Token costs were last updated on {self.token_cost_date} and may have changed since then."
        )
        assert (
            tiktoken_model_name in self.cost_mapping.keys()
        ), f"Only {list(self.cost_mapping.keys())} are supported"

        print(f"langfair: Estimating cost based on {count} generations per prompt...")

        if example_responses is None:
            print("langfair: Generating sample of responses for cost estimation...")
            prompts = list(prompts)
            sampled_prompts = random.sample(
                prompts, min(response_sample_size, len(prompts))
            )
            generation = await self.generate_responses(sampled_prompts, count=1)
            example_responses = generation["data"]["response"]

        # Get input token counts
        prompt_token_counts = [
            self._num_tokens_from_messages(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                model=tiktoken_model_name,
            )
            for user_prompt in prompts
        ]
        total_prompt_tokens = sum(prompt_token_counts) * count

        # Estimate output token counts
        example_response_tokens = [
            self._num_tokens_from_messages(
                [{"role": "assistant", "content": response}],
                model=tiktoken_model_name,
                prompt=False,
            )
            for response in example_responses
        ]
        estimated_total_response_tokens = (
            len(prompts) * np.mean(example_response_tokens) * count
        )

        # calculate costs
        model_cost = self.cost_mapping.get(
            tiktoken_model_name, {"input": 0, "output": 0}
        )
        estimated_prompt_token_cost = total_prompt_tokens * model_cost["input"]
        estimated_completion_token_cost = (
            estimated_total_response_tokens * model_cost["output"]
        )
        estimated_total_token_cost = (
            estimated_prompt_token_cost + estimated_completion_token_cost
        )

        results = {
            "Estimated Prompt Token Cost (USD)": estimated_prompt_token_cost,
            "Estimated Completion Token Cost (USD)": estimated_completion_token_cost,
            "Estimated Total Token Cost (USD)": estimated_total_token_cost,
        }
        return results

    async def generate_responses(
        self,
        prompts: List[str],
        system_prompt: str = "You are a helpful assistant.",
        count: int = 25,
    ) -> Dict[str, Any]:
        """
        Generates evaluation dataset from a provided set of prompts. For each prompt,
        `self.count` responses are generated.

        Parameters
        ----------
        prompts : list of strings
            List of prompts from which LLM responses will be generated

        system_prompt : str or None, default="You are a helpful assistant."
            Optional argument for user to provide custom system prompt

        count : int, default=25
            Specifies number of responses to generate for each prompt. The convention is to use 25
            generations per prompt in evaluating toxicity. See, for example DecodingTrust (https://arxiv.org/abs//2306.11698)
            or Gehman et al., 2020 (https://aclanthology.org/2020.findings-emnlp.301/).

        Returns
        -------
        dict
            A dictionary with two keys: 'data' and 'metadata'.
            'data' : dict
                A dictionary containing the prompts and responses.
                'prompt' : list
                    A list of prompts.
                'response' : list
                    A list of responses corresponding to the prompts.
            'metadata' : dict
                A dictionary containing metadata about the generation process.
                'non_completion_rate' : float
                    The rate at which the generation process did not complete.
                'temperature' : float
                    The temperature parameter used in the generation process.
                'count' : int
                    The count of prompts used in the generation process.
                'system_prompt' : str
                    The system prompt used for generating responses
        """
        assert isinstance(self.llm, langchain_core.runnables.base.Runnable), """
            langchain_llm must be an instance of langchain_core.runnables.base.Runnable
        """
        assert all(
            isinstance(prompt, str) for prompt in prompts
        ), "langfair: If using custom prompts, please ensure `prompts` is of type list[str]"
        print(f"langfair: Generating {count} responses per prompt...")
        if self.llm.temperature == 0:
            assert (
                count == 1
            ), "langfair: temperature must be greater than 0 if count > 1"
        self._update_count(count)

        # set up langchain and generate asynchronously
        chain = self._setup_langchain(system_message=system_prompt)
        generations, duplicated_prompts = await self._generate_in_batches(
            chain=chain, prompts=prompts
        )
        responses = []
        for response in generations:
            responses.extend(response)

        non_completion_rate = len([r for r in responses if r == FAILURE_MESSAGE]) / len(
            responses
        )

        print("langfair: Responses successfully generated!")
        return {
            "data": {
                "prompt": duplicated_prompts,
                "response": responses,
            },
            "metadata": {
                "non_completion_rate": non_completion_rate,
                "system_prompt": system_prompt,
                "temperature": self.llm.temperature,
                "count": self.count,
            },
        }

    def _setup_langchain(self, system_message: str = "{system_text}") -> Any:
        """Sets up langchain `LLMChain` object"""
        system_msg_prompt = SystemMessagePromptTemplate.from_template(system_message)
        human_prompt = HumanMessagePromptTemplate.from_template("{text}")
        messages = [system_msg_prompt, human_prompt]
        chat_prompt = ChatPromptTemplate.from_messages(messages)
        chain = LLMChain(llm=self.llm, prompt=chat_prompt)
        return chain

    def _task_creator(
        self,
        chain: Any,
        prompts: List[str],
        system_prompts: Optional[List[str]] = None,
    ) -> Tuple[List[Any], List[str]]:
        """
        Creates a list of async tasks and returns duplicated prompt list
        with each prompt duplicated `count` times
        """
        duplicated_prompts = [
            prompt for prompt, i in itertools.product(prompts, range(self.count))
        ]
        # Use `n` parameter if instance of AzureChatOpenAI
        if isinstance(self.llm, langchain_openai.chat_models.azure.AzureChatOpenAI):
            system_prompts = (
                [None] * len(prompts) if system_prompts is None else system_prompts
            )
            tasks = [
                self._async_api_call(
                    chain=chain,
                    prompt=prompt,
                    count=self.count,
                    system_text=system_prompt,
                )
                for prompt, system_prompt in zip(prompts, system_prompts)
            ]

        # Do not use `n` parameter otherwise
        else:
            if not system_prompts:
                system_prompts = [None] * len(duplicated_prompts)
            else:
                system_prompts = [
                    val
                    for val, i in itertools.product(system_prompts, range(self.count))
                ]
            tasks = [
                self._async_api_call(
                    chain=chain, prompt=prompt, count=1, system_text=system_prompt
                )
                for prompt, system_prompt in zip(duplicated_prompts, system_prompts)
            ]
        return tasks, duplicated_prompts

    def _update_count(self, count: int) -> None:
        """Updates self.count parameter and self.llm as necessary"""
        self.count = count
        self.llm.n = (
            count
            if isinstance(self.llm, langchain_openai.chat_models.azure.AzureChatOpenAI)
            else 1
        )

    async def _generate_in_batches(
        self,
        chain: Any,
        prompts: List[str],
        system_prompts: Optional[List[str]] = None,
    ) -> Tuple[List[str], List[str]]:
        """Executes async IO with langchain in batches to avoid rate limit error"""
        # define batch size and partition prompt list
        batch_size = (
            len(prompts)
            if not self.max_calls_per_min
            else self.max_calls_per_min // self.count
        )
        prompts_partition = self._split(prompts, batch_size)

        # Execute async in batches
        duplicated_prompts, responses = [], []
        for prompt_batch in prompts_partition:
            start = time.time()
            # generate responses for current batch
            tasks, duplicated_batch_prompts = self._task_creator(
                chain, prompt_batch, system_prompts
            )
            responses_batch = await asyncio.gather(*tasks)

            # extend lists to include current batch
            duplicated_prompts.extend(duplicated_batch_prompts)
            responses.extend(responses_batch)
            stop = time.time()

            # pause if needed
            if (stop - start < 60) and (batch_size < len(prompts)):
                time.sleep(61 - stop + start)

        return responses, duplicated_prompts

    ################################################################################
    # Helper function for OpenAI API calls
    ################################################################################
    @staticmethod
    async def _async_api_call(
        chain: Any, prompt: str, system_text: Optional[str] = None, count: int = 1
    ) -> List[Any]:
        """Generates responses asynchronously using an LLMChain object"""
        try:
            result = await chain.agenerate(
                [{"text": prompt, "system_text": system_text}]
            )
            return [result.generations[0][i].text for i in range(count)]
        except (
            openai.APIConnectionError,
            openai.NotFoundError,
            openai.InternalServerError,
            openai.PermissionDeniedError,
            openai.AuthenticationError,
            openai.RateLimitError,
        ):
            raise
        except Exception:
            return [FAILURE_MESSAGE] * count

    @staticmethod
    def _split(list_a: List[str], chunk_size: int) -> List[List[str]]:
        """Partitions list"""
        for i in range(0, len(list_a), chunk_size):
            yield list_a[i : i + chunk_size]

    @staticmethod
    def _num_tokens_from_messages(
        messages: List[Dict[str, str]], model: str, prompt: bool = True
    ) -> int:
        """
        Returns the number of tokens used by a list of messages.

        Note : This code is adapted from the `openai-cookbook` GitHub repository.
        Source: https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
        """
        model_data = {
            "gpt-3.5-turbo-0301": (4, 1),
            "gpt-3.5-turbo-0613": (3, 1),
            "gpt-3.5-turbo-16k-0613": (3, 1),
            "gpt-4-0314": (3, 1),
            "gpt-4-32k-0314": (3, 1),
            "gpt-4-0613": (3, 1),
            "gpt-4-32k-0613": (3, 1),
        }
        if model not in model_data:
            if "gpt-3.5-turbo" in model:
                print(
                    "Warning: gpt-3.5-turbo may update over time. Returning num tokens assuming gpt-3.5-turbo-0613."
                )
                model = "gpt-3.5-turbo-0613"
            elif "gpt-4" in model:
                print(
                    "Warning: gpt-4 may update over time. Returning num tokens assuming gpt-4-0613."
                )
                model = "gpt-4-0613"
            else:
                raise NotImplementedError(
                    f"""cost_estimator() is not implemented for model {model}."""
                )
        tokens_per_message, tokens_per_name = model_data[model]
        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            print("Warning: model not found. Using cl100k_base encoding.")
            encoding = tiktoken.get_encoding("cl100k_base")
        num_tokens = 0
        for message in messages:
            if prompt:
                num_tokens += tokens_per_message
            for key, value in message.items():
                num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
        if prompt:
            num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
        elif not prompt:
            num_tokens += -1
        return num_tokens
