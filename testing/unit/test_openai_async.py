import asyncio
from arc_agi.patterns.find_patterns import get_completion


async def test_async_completions():
    """Test async completion functionality"""
    prompts = ["Hello, how are you?", "Hey, can you explain me Google?"]
    tasks = [get_completion(p) for p in prompts]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    print("Results:")
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            print(f"Prompt {i+1} failed: {result}")
        else:
            print(f"Prompt {i+1}: {result[:100]}..." if len(str(result)) > 100 else f"Prompt {i+1}: {result}")
    
    return results


if __name__ == "__main__":
    # Run the async test
    asyncio.run(test_async_completions())