import langfair
from langfair.metrics.toxicity import ToxicityMetrics
import asyncio
import grpc

print("✅ LangFair is installed!")
print(f"🔹 LangFair Version: {langfair.__version__}\n")

# ------------------- TEST 1: TOXICITY EVALUATION -------------------
print("🧪 Running Toxicity Evaluation...")

tm = ToxicityMetrics()
test_prompts = ["Hello, how are you?", "You are stupid."]
test_responses = ["I'm fine, thank you!", "That was rude."]

tox_result = tm.evaluate(prompts=test_prompts, responses=test_responses, return_data=True)

print("✅ Toxicity Metrics:", tox_result['metrics'])
print("\n----------------------------------------------------\n")

# ------------------- TEST 2: LLM RESPONSE GENERATION (OPTIONAL) -------------------
from langchain_google_vertexai import ChatVertexAI
from langfair.generator import ResponseGenerator
from langchain_core.rate_limiters import InMemoryRateLimiter

print("🧪 Running LLM Response Generation...")

rate_limiter = InMemoryRateLimiter(requests_per_second=4.5, check_every_n_seconds=0.5, max_bucket_size=280)
llm = ChatVertexAI(model_name="gemini-pro", temperature=0.3, rate_limiter=rate_limiter)

rg = ResponseGenerator(langchain_llm=llm)

async def generate_llm_responses():
    test_generations = await rg.generate_responses(prompts=["Tell me a joke."], count=3)
    print("✅ Generated Responses:", test_generations["data"]["response"])

if __name__ == "__main__":
    try:
        asyncio.run(generate_llm_responses())
        print("\n🎉 All tests completed successfully!")
    finally:
        # Remove the explicit gRPC shutdown call entirely
        pass  # gRPC client handles cleanup automatically
