from arc_agi.src.solver.solver_prompts import example_template, solver_prompt_template, new_solver_prompt, critic_prompt
from arc_agi.src.utils.llm_utils import get_anthropic_response_stream, get_cerebras_response, parse_grid, call_llm
from arc_agi.src.utils.visualization_utils import get_arr_viz
from arc_agi.src.objects.base import Grid, BaseObject
import re, asyncio, json

def create_base_object_sync(grid, coord_tuples):
    """
    Helper: given a 2D list (or array) and a set of (x, y) tuples,
    construct a BaseObject synchronously.
    """
    # BaseObject constructor expects Set[Tuple[int, int]] directly
    data = BaseObject(grid, coord_tuples).to_dict(
        provider="openai", 
        model="gpt-4.1-mini", 
        temperature=0.0, 
        max_tokens=4096
    )
    del data["grid"]
    return data

async def create_base_object(grid, coord_tuples):
    """
    Async wrapper that runs the sync function in a thread pool
    """
    return await asyncio.to_thread(create_base_object_sync, grid, coord_tuples)

def get_formatted_examples(example_inputs):
    examples_str = ""
    
    for i, entry in enumerate(example_inputs):
        input_viz = get_arr_viz(entry["input"])
        output_viz = get_arr_viz(entry["output"])
        examples_str += example_template.format(i=i+1, input_viz=input_viz, output_viz=output_viz) + "\n"
        
    return examples_str.strip()

async def get_objects(grid_input):

    grid_a = Grid(grid_input)
    input_obj = grid_a.find_objects_in_grid('openai', 'gpt-4.1-mini', 0.0, 4096)
    #input_obj = grid_a.objects_concatenator('openai', 'gpt-4.1-mini', 0.0, 4096)
    input_tasks = [create_base_object(grid_input, obj) for obj in input_obj]
    before_list = await asyncio.gather(*input_tasks)
    return before_list

async def get_prompts(arc_input, hint):
    examples_str = get_formatted_examples(arc_input["train"])
    
    # Get objects for all test inputs concurrently (not sequentially!)
    print("Finding the Objects")
    
    # Create all object extraction tasks concurrently
    #object_tasks = [get_objects(entry['input']) for entry in arc_input['test']]
    test_input_vizs = [get_arr_viz(entry['input']) for entry in arc_input['test']]
    
    # Wait for all object extractions to complete concurrently
    #all_objects = await asyncio.gather(*object_tasks)
    
    # Combine visualization and objects
    #test_objects = [(viz, json.dumps(objects)) for viz, objects in zip(test_input_vizs, all_objects)]
    
    prompt_arr = [new_solver_prompt.format(examples=examples_str, test_input_viz=test_input_viz, hint=hint) for test_input_viz in test_input_vizs]
    ground_truths_arr = [entry.get('output', []) for entry in arc_input['test']]
    
    return prompt_arr, ground_truths_arr

def extract_matrix_from_response(response):
    # Extract text between backticks (```)
    code_blocks = re.findall(r'```(.*?)```', response, re.DOTALL)
    
    if code_blocks:
        # Return the first code block found, stripped of whitespace
        return code_blocks[0].strip()
    
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
        row = [int(x) for x in line.split("|") if x.strip()]
        if row:  # Only add non-empty rows
            arr.append(row)
    return arr

async def get_solved_outputs(arc_input, hint, return_raw_responses=False):
    print("Creating the Prompt")
    prompt_arr, ground_truths_arr = await get_prompts(arc_input, hint)
    #raw_responses = [get_anthropic_response_stream(prompt) for prompt in prompt_arr]
    print("Started the LLM call")
    raw_responses = [get_anthropic_response_stream(prompt) for prompt in prompt_arr]
    print(raw_responses)
    critic_prompts = []
    for i in range(len(raw_responses)):
        critic_prompts.append(critic_prompt.format(prompt_arr[i],raw_responses[i]))
    raw_responses = [get_anthropic_response_stream(prompt) for prompt in critic_prompts]
    print(raw_responses)
    responses = [extract_matrix_from_response(response) for response in raw_responses]
    arr_responses = [matrix_to_arr(response) for response in responses]
    critic_prompts = []
    for i in range(len(raw_responses)):
        critic_prompts.append(critic_prompt.format(get_arr_viz(arr_responses[i]),raw_responses[i]))
    raw_responses = [get_anthropic_response_stream(prompt) for prompt in critic_prompts]
    print(raw_responses)
    
    responses = [extract_matrix_from_response(response) for response in raw_responses]
    arr_responses = [matrix_to_arr(response) for response in responses]
    if return_raw_responses:
        return (raw_responses, arr_responses), ground_truths_arr
    else:
        return arr_responses, ground_truths_arr

if __name__ == "__main__":
    async def main():
        # This is taken from the evaluation folder from the official ARC AGI 2 repo: data/evaluation/581f7754.json
        arc_input = {"train": [{"input": [[1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1], [1, 8, 8, 8, 1, 1, 1, 1], [1, 8, 4, 8, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 8, 1], [1, 1, 1, 1, 8, 8, 4, 1], [1, 1, 1, 1, 1, 1, 8, 1], [1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1], [1, 8, 1, 8, 1, 1, 1, 1], [1, 8, 4, 8, 1, 1, 1, 1], [1, 8, 1, 8, 1, 1, 1, 1], [1, 8, 8, 8, 1, 1, 1, 1], [1, 1, 1, 1, 1, 4, 1, 1]], "output": [[1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 8, 8, 8, 1], [1, 1, 1, 1, 8, 4, 8, 1], [1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 8, 1, 1], [1, 1, 1, 8, 8, 4, 1, 1], [1, 1, 1, 1, 1, 8, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 8, 1, 8, 1], [1, 1, 1, 1, 8, 4, 8, 1], [1, 1, 1, 1, 8, 1, 8, 1], [1, 1, 1, 1, 8, 8, 8, 1], [1, 1, 1, 1, 1, 4, 1, 1]]}, {"input": [[8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8], [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8], [8, 8, 4, 8, 8, 8, 8, 8, 8, 3, 3, 3, 8, 8, 8, 8, 8, 8, 8, 8, 8], [8, 8, 4, 8, 8, 8, 8, 8, 8, 3, 1, 3, 8, 8, 8, 8, 8, 8, 8, 8, 8], [1, 8, 4, 8, 3, 3, 3, 8, 8, 3, 3, 3, 8, 8, 8, 8, 8, 8, 8, 8, 8], [8, 8, 4, 8, 3, 8, 3, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 3, 8, 8, 8], [8, 8, 6, 8, 3, 8, 3, 8, 8, 8, 8, 8, 8, 8, 3, 1, 3, 3, 8, 8, 8], [8, 8, 8, 8, 3, 1, 3, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 3, 8, 8, 8], [6, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8], [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8]], "output": [[8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8], [8, 8, 8, 8, 3, 3, 3, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8], [8, 8, 8, 8, 3, 8, 3, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8], [8, 8, 8, 8, 3, 8, 3, 8, 3, 3, 3, 8, 8, 8, 8, 8, 8, 3, 8, 8, 8], [1, 8, 4, 8, 3, 1, 3, 8, 3, 1, 3, 8, 8, 8, 3, 1, 3, 3, 8, 8, 8], [8, 8, 4, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 3, 8, 8, 8], [8, 8, 4, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8], [8, 8, 4, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8], [6, 8, 6, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8], [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8]]}, {"input": [[3, 3, 3, 3, 2, 3, 3, 3, 3, 3, 3], [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], [3, 1, 1, 1, 3, 3, 3, 3, 3, 3, 3], [3, 3, 1, 3, 3, 3, 3, 3, 3, 3, 3], [3, 3, 2, 3, 3, 3, 3, 3, 3, 3, 3], [3, 1, 1, 1, 3, 3, 3, 3, 3, 3, 3], [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], [3, 3, 3, 1, 1, 1, 1, 1, 3, 3, 3], [3, 3, 3, 1, 3, 3, 3, 1, 3, 3, 3], [3, 3, 3, 1, 1, 1, 2, 1, 3, 3, 3], [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], [3, 3, 3, 3, 1, 1, 1, 3, 3, 3, 3], [3, 3, 3, 1, 1, 2, 1, 1, 3, 3, 3], [3, 3, 3, 3, 3, 1, 3, 3, 3, 3, 3], [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]], "output": [[3, 3, 3, 3, 2, 3, 3, 3, 3, 3, 3], [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], [3, 3, 3, 1, 1, 1, 3, 3, 3, 3, 3], [3, 3, 3, 3, 1, 3, 3, 3, 3, 3, 3], [3, 3, 3, 3, 2, 3, 3, 3, 3, 3, 3], [3, 3, 3, 1, 1, 1, 3, 3, 3, 3, 3], [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], [3, 1, 1, 1, 1, 1, 3, 3, 3, 3, 3], [3, 1, 3, 3, 3, 1, 3, 3, 3, 3, 3], [3, 1, 1, 1, 2, 1, 3, 3, 3, 3, 3], [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], [3, 3, 3, 1, 1, 1, 3, 3, 3, 3, 3], [3, 3, 1, 1, 2, 1, 1, 3, 3, 3, 3], [3, 3, 3, 3, 1, 3, 3, 3, 3, 3, 3], [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]], "output": [[3, 3, 3, 3, 2, 3, 3, 3, 3, 3, 3], [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], [3, 3, 3, 1, 1, 1, 3, 3, 3, 3, 3], [3, 3, 3, 3, 1, 3, 3, 3, 3, 3, 3], [3, 3, 3, 3, 2, 3, 3, 3, 3, 3, 3], [3, 3, 3, 1, 1, 1, 3, 3, 3, 3, 3], [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], [3, 1, 1, 1, 1, 1, 3, 3, 3, 3, 3], [3, 1, 3, 3, 3, 1, 3, 3, 3, 3, 3], [3, 1, 1, 1, 2, 1, 3, 3, 3, 3, 3], [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], [3, 3, 3, 1, 1, 1, 3, 3, 3, 3, 3], [3, 3, 1, 1, 2, 1, 1, 3, 3, 3, 3], [3, 3, 3, 3, 1, 3, 3, 3, 3, 3, 3], [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]]}], "test": [{"input": [[8, 8, 8, 6, 6, 6, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8], [8, 8, 8, 6, 4, 6, 8, 8, 8, 8, 8, 8, 8, 4, 6, 4, 8, 8, 8, 8], [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 4, 4, 4, 8, 8, 8, 8], [6, 8, 8, 8, 8, 8, 2, 8, 8, 8, 8, 8, 8, 8, 4, 8, 8, 8, 8, 8], [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8], [8, 8, 8, 8, 4, 8, 8, 2, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8], [8, 8, 8, 4, 4, 4, 8, 8, 8, 8, 4, 8, 8, 8, 8, 8, 8, 3, 3, 3], [8, 2, 8, 4, 8, 4, 8, 8, 4, 4, 4, 8, 8, 8, 8, 8, 8, 3, 8, 3], [8, 8, 8, 4, 4, 4, 8, 8, 4, 4, 6, 8, 8, 8, 8, 8, 8, 3, 8, 3], [8, 8, 8, 8, 6, 8, 8, 8, 8, 8, 4, 8, 8, 8, 8, 8, 8, 3, 6, 3], [4, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 2, 8, 8, 8, 8, 8, 8, 8], [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8], [2, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 2], [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8]], "output": [[8, 8, 8, 4, 4, 4, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 3, 3, 3], [8, 8, 8, 4, 8, 4, 8, 8, 8, 8, 4, 8, 8, 8, 8, 8, 8, 3, 8, 3], [8, 8, 8, 4, 4, 4, 8, 8, 4, 4, 4, 8, 8, 8, 8, 8, 8, 3, 8, 3], [6, 8, 8, 8, 6, 8, 8, 8, 4, 4, 6, 8, 8, 4, 6, 4, 8, 3, 6, 3], [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 4, 8, 8, 4, 4, 4, 8, 8, 8, 8], [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 4, 8, 8, 8, 8, 8], [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8], [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8], [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8], [8, 8, 8, 6, 6, 6, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8], [4, 8, 8, 6, 4, 6, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8], [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8], [2, 2, 8, 8, 8, 8, 2, 2, 8, 8, 8, 8, 2, 8, 8, 8, 8, 8, 8, 2], [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8]]}, {"input": [[4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 1, 4, 4, 4, 4, 4, 4, 4, 4], [4, 4, 4, 1, 1, 1, 4, 4, 4, 4, 4, 4, 3, 1, 4, 4, 4, 4, 4, 4, 4, 4], [4, 1, 1, 1, 4, 1, 4, 4, 4, 4, 4, 4, 4, 1, 4, 4, 4, 4, 4, 4, 4, 4], [4, 4, 4, 1, 4, 1, 1, 1, 4, 4, 4, 4, 4, 1, 4, 4, 4, 4, 4, 4, 4, 2], [4, 4, 4, 1, 2, 1, 4, 4, 4, 4, 4, 4, 4, 1, 4, 4, 4, 1, 4, 4, 4, 4], [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 1, 4, 1, 1, 1, 4, 4, 4, 4], [4, 4, 4, 4, 4, 4, 4, 4, 4, 1, 2, 1, 4, 8, 4, 1, 4, 1, 4, 4, 4, 4], [4, 4, 4, 4, 4, 4, 4, 4, 4, 1, 4, 1, 4, 4, 4, 2, 4, 1, 4, 4, 4, 4], [4, 4, 4, 4, 4, 4, 4, 4, 4, 1, 1, 1, 4, 4, 4, 4, 4, 4, 4, 4, 4, 8], [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4], [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4]], "output": [[4, 4, 4, 1, 1, 1, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 1, 4, 4, 4, 4], [4, 1, 1, 1, 4, 1, 4, 4, 4, 4, 4, 4, 4, 4, 4, 1, 1, 1, 4, 4, 4, 4], [4, 4, 4, 1, 4, 1, 1, 1, 4, 4, 4, 4, 3, 1, 4, 1, 4, 1, 4, 4, 4, 4], [4, 4, 4, 1, 2, 1, 4, 4, 4, 1, 2, 1, 3, 1, 4, 2, 4, 1, 4, 4, 4, 2], [4, 4, 4, 4, 4, 4, 4, 4, 4, 1, 4, 1, 4, 1, 4, 4, 4, 4, 4, 4, 4, 4], [4, 4, 4, 4, 4, 4, 4, 4, 4, 1, 1, 1, 4, 1, 4, 4, 4, 4, 4, 4, 4, 4], [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 1, 4, 4, 4, 4, 4, 4, 4, 4], [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 1, 4, 4, 4, 4, 4, 4, 4, 4], [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 8, 4, 4, 4, 4, 4, 4, 4, 8], [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4], [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4]]}]}
        
        # This is the hint that I gave for testing purposes.
        hint = """There seem to be rectangular regions with a set of digits within (let's call that content of the object). And near the bottom, there seem to be unit digits. 
    You need to connect the content of each object to the content of the other objects in the order of the unit digits at the bottom. The color (digit) of the connections should be the same as the color of the content that the connection is coming from.
    Examples are given for your reference"""
        
        #prompt_arr, ground_truths_arr = await get_prompts(arc_input, hint)

        # res = get_anthropic_response_stream(prompt_arr[0])
        
        #responses, arr_responses = await get_solved_outputs(arc_input, hint)
        
        #print(prompt_arr[0])
        #print(arr_responses[0])
    
    #asyncio.run(main())
