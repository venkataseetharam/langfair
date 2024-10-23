import pytest
from langchain_openai import AzureChatOpenAI

from langfair.generator import ResponseGenerator


@pytest.mark.asyncio
async def test_generator(monkeypatch):
    count = 3
    MOCKED_PROMPTS = ["Prompt 1", "Prompt 2", "Prompt 3"]
    MOCKED_DUPLICATE_PROMPTS = ["Prompt 1", "Prompt 2", "Prompt 3"] * count
    MOCKED_RESPONSES = [
        "Mocked response 1",
        "Mocked response 2",
        "Unable to get response",
    ] * count

    async def mock_generate_in_batches(*args, **kwargs):
        return [MOCKED_RESPONSES], MOCKED_DUPLICATE_PROMPTS

    mock_object = AzureChatOpenAI(
        deployment_name="YOUR-DEPLOYMENT",
        temperature=1,
        api_key="SECRET_API_KEY",
        api_version="2024-05-01-preview",
        azure_endpoint="https://mocked.endpoint.com",
    )

    generator_object = ResponseGenerator(langchain_llm=mock_object)

    monkeypatch.setattr(
        generator_object, "_generate_in_batches", mock_generate_in_batches
    )
    data = await generator_object.generate_responses(prompts=MOCKED_PROMPTS)

    cost = await generator_object.estimate_token_cost(
        tiktoken_model_name="gpt-3.5-turbo-16k-0613",
        prompts=MOCKED_DUPLICATE_PROMPTS,
        example_responses=MOCKED_RESPONSES[:3],
        count=count,
    )

    assert data["data"]["response"] == MOCKED_RESPONSES
    assert data["metadata"]["non_completion_rate"] == 1 / 3
    assert cost == {
        "Estimated Prompt Token Cost (USD)": 0.001539,
        "Estimated Completion Token Cost (USD)": 0.000504,
        "Estimated Total Token Cost (USD)": 0.002043,
    }
