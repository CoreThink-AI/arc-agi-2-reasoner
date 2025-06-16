import anthropic

anthropic_client = anthropic.Anthropic()

def get_anthropic_response(prompt):
    response = anthropic_client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=16000,
        thinking={
            "type": "enabled",
            "budget_tokens": 10000
        },
        messages=[{
            "role": "user",
            "content": prompt
        }]
    )

    # The response will contain summarized thinking blocks and text blocks
    for block in response.content:
        if block.type == "thinking":
            print(f"\nThinking summary: {block.thinking}")
        elif block.type == "text":
            print(f"\nResponse: {block.text}")

def get_anthropic_response_stream(prompt):
    with anthropic_client.messages.stream(
        model="claude-opus-4-20250514",
        max_tokens=16000,
        thinking={
            "type": "enabled",
            "budget_tokens": 10000
        },
        messages=[{
            "role": "user",
            "content": prompt
        }]
    ) as stream:
        for text in stream.text_stream:
            print(text, end="", flush=True)