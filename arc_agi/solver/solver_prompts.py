example_template = """Example {i}:
Input:
{input_viz}

Output:
{output_viz}
"""

prompt_template = """Consider the following examples:

{examples}

Based on the pattern in the examples, what would the output for the following test input be?

Test input:
{test_input_viz}

Test output:

(Hint: {hint}) 
"""