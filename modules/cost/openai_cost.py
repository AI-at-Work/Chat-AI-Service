def estimate_llm_api_cost(model, num_tokens_input, num_tokens_output):
    # Prices of per thousand tokens
    pricing = {
        "llama3.1:8b": {"input": 0.00150, "output": 0.00200},
        "gpt-3.5-turbo-0125": {"input": 0.00050, "output": 0.00150},
        "gpt-3.5-turbo": {"input": 0.0030, "output": 0.0060},
        "gpt-3.5-turbo-1106": {"input": 0.0010, "output": 0.0020},
        "gpt-3.5-turbo-instruct": {"input": 0.00150, "output": 0.00200},
        "gpt-4-turbo": {"input": 0.0100, "output": 0.0300},
        "gpt-4-turbo-2024-04-09": {"input": 0.0100, "output": 0.0300},
        "gpt-4": {"input": 0.0300, "output": 0.0600},
        "gpt-4-32k": {"input": 0.0600, "output": 0.1200},
        "text-embedding-3-small": {"input": 0.00002, "output": 0.0},
        "text-embedding-3-large": {"input": 0.00013, "output": 0.0},
        "text-embedding-ada-002": {"input": 0.00010, "output": 0.0},
    }

    if model not in pricing:
        raise ValueError(f"Unknown model: {model}")

    input_cost = (num_tokens_input / 1000) * pricing[model]["input"]
    output_cost = (num_tokens_output / 1000) * pricing[model]["output"]
    total_cost = input_cost + output_cost

    return round(total_cost, 6)


if __name__ == "__main__":

    # Example usage
    model = "gpt-3.5-turbo"
    num_tokens_input = 1000
    num_tokens_output = 1000

    estimated_cost = estimate_llm_api_cost(model, num_tokens_input, num_tokens_output)
    print(f"Estimated cost for input: {num_tokens_input} and output: {num_tokens_output} tokens using {model}: ${estimated_cost}")