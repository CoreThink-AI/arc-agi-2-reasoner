import re
import os
import json

original_prompt_template = """Below is a list of input and output pairs with a pattern. Your goal is to identify the pattern or transformation in the training examples that maps the input to the output, then apply that pattern to the test input to give a final output.

{examples}


Below is the test input grid. Predict the corresponding output grid by applying the rule you found. Respond in the format of the training output examples.  Your final answer should just be the text output grid itself. 

Input:
{test_input_viz}

Enter the output grid within ``` and ```.
"""

example_template = """Example {i}:

Input:
```
{input_viz}
```
Output:
```
{output_viz}
```
"""


def get_arr_viz(arr):
    viz = ""
    for row in arr:
        viz += " ".join(str(i) for i in row) + "\n"

    return viz.strip()


def get_formatted_examples(example_inputs, objects_train):
    examples_str = ""

    for i, entry in enumerate(example_inputs):
        input_viz = get_arr_viz(entry["input"])
        # input_objects = objects_train[i][1]
        output_viz = get_arr_viz(entry["output"])
        # output_objects = objects_train[i][2]
        # examples_str += example_template.format(i=i + 1, input_viz=input_viz, output_viz=output_viz, input_objects=input_objects, output_objects=output_objects) + "\n"
        examples_str += example_template.format(i=i + 1, input_viz=input_viz, output_viz=output_viz) + "\n"

    return examples_str.strip()


def get_prompts(task_json, objects_train):
    arc_input = task_json
    examples_str = get_formatted_examples(arc_input["train"], objects_train)
    test_input_viz_arr = [get_arr_viz(entry['input']) for entry in arc_input['test']]

    prompt_arr = [original_prompt_template.format(examples=examples_str, test_input_viz=test_input_viz) for
                  test_input_viz in test_input_viz_arr]

    return prompt_arr


def append_to_existing_json(entry, file_path, key=None):
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # Load existing data or create new structure
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            existing_data = json.load(f)
    else:
        existing_data = {} if key else []

    # Append the entry
    if key:
        # If key is provided, append to that key (creates list if it doesn't exist)
        if key not in existing_data:
            existing_data[key] = []
        existing_data[key].append(entry)
    else:
        # If no key, assume we're appending to a list
        if not isinstance(existing_data, list):
            # Convert to list if it's not already
            existing_data = [existing_data]
        existing_data.append(entry)

    # Save back to file
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(existing_data, f, indent=2, ensure_ascii=False)


def extract_matrix_from_response(response):
    # Extract text between backticks (```)
    code_blocks = re.findall(r'```(.*?)```', response, re.DOTALL)

    if code_blocks:
        # Return the first code block found, stripped of whitespace
        return code_blocks[-1].strip()

    # Fallback: try to extract matrix-like content with square brackets
    matrix = re.search(r'\[(.*?)\]', response, re.DOTALL)
    if matrix:
        return matrix.group(0)

    return ""


def matrix_to_arr(matrix_str):
    matrix_str = matrix_str.replace("```", "")
    lines = matrix_str.split("\n")
    arr = []
    for line in lines:
        line = line.strip()
        # Filter out empty strings before converting to int
        row = [int(x) for x in line.split(" ") if x.strip()]
        if row:  # Only add non-empty rows
            arr.append(row)
    return arr
