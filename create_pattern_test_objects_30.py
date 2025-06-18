import json
import os
from collections import Counter
from arc_agi.src.objects.object_finder import find_objects

def find_background_color_simple(grid):
    """Simple rule-based background color detection without LLM"""
    # Flatten the grid and count color frequencies
    flat_grid = [cell for row in grid for cell in row]
    color_counts = Counter(flat_grid)
    
    # Get edge colors (border cells)
    edge_colors = []
    height, width = len(grid), len(grid[0])
    
    # Top and bottom edges
    for j in range(width):
        edge_colors.append(grid[0][j])  # Top edge
        edge_colors.append(grid[height-1][j])  # Bottom edge
    
    # Left and right edges (excluding corners to avoid double counting)
    for i in range(1, height-1):
        edge_colors.append(grid[i][0])  # Left edge
        edge_colors.append(grid[i][width-1])  # Right edge
    
    edge_color_counts = Counter(edge_colors)
    
    # Find most common color that also appears on edges
    most_common_color = color_counts.most_common(1)[0][0]
    most_common_edge_color = edge_color_counts.most_common(1)[0][0] if edge_color_counts else most_common_color
    
    # If the most common color is also the most common edge color, it's likely the background
    if most_common_color == most_common_edge_color:
        return most_common_color
    
    # Otherwise, prefer the most common color
    return most_common_color

def create_pattern_testing_objects():
    """Create pattern testing objects for the specified 31 task IDs"""
    
    # List of task IDs to process
    task_ids = [
        "7b80bb43", "7c66cb00", "7ed72f31", "800d221b", "80a900e0", "88bcf3b4", 
        "88e364bc", "8b7bacbf", "8b9c3697", "8e5c0c38", "8f215267", "8f3a5a89", 
        "9385bd28", "97d7923e", "981571dc", "9aaea919", "9bbf930d", "a25697e4", 
        "a395ee82", "a47bf94d", "aa4ec2a5", "b10624e5", "b5ca7ac4", "b6f77b65", 
        "b99e7126", "b9e38dc0", "c4d067a0", "c7f57c3e", "cb2d8a2c", "cbebaa4b", 
        "d35bdbdc"
    ]
    
    result = {}
    data_dir = "data"
    
    print(f"Processing {len(task_ids)} task IDs...")
    
    for i, task_id in enumerate(task_ids, 1):
        print(f"Processing {i}/{len(task_ids)}: {task_id}")
        
        task_file = os.path.join(data_dir, f"{task_id}.json")
        
        if not os.path.exists(task_file):
            print(f"Warning: File {task_file} not found, skipping...")
            continue
            
        try:
            with open(task_file, 'r') as f:
                task_data = json.load(f)
            
            result[task_id] = {"train": []}
            
            # Process each training example
            for example in task_data["train"]:
                input_grid = example["input"]
                output_grid = example["output"]
                
                # Find background colors
                input_bg = find_background_color_simple(input_grid)
                output_bg = find_background_color_simple(output_grid)
                
                # Find objects in input and output
                input_objects = find_objects(input_grid, input_bg)
                output_objects = find_objects(output_grid, output_bg)
                
                # Convert coordinate sets to lists of [x,y] pairs
                input_coords = []
                for obj in input_objects:
                    obj_coords = [list(coord) for coord in obj]
                    input_coords.append(obj_coords)
                
                output_coords = []
                for obj in output_objects:
                    obj_coords = [list(coord) for coord in obj]
                    output_coords.append(obj_coords)
                
                result[task_id]["train"].append({
                    "input": input_coords,
                    "output": output_coords
                })
                
        except Exception as e:
            print(f"Error processing {task_id}: {e}")
            continue
    
    # Save the result
    output_file = "testing/samples/pattern_test_objects_30.json"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Custom JSON formatting to match the original structure
    with open(output_file, "w") as f:
        f.write("{\n")
        task_items = list(result.items())
        for i, (task_id, task_data) in enumerate(task_items):
            f.write(f'  "{task_id}": {{\n')
            f.write('    "train": [\n')
            
            for j, example in enumerate(task_data["train"]):
                f.write('      {\n')
                
                # Write input
                f.write('        "input": [')
                input_strs = []
                for obj in example["input"]:
                    obj_str = "[" + ",".join(f"[{coord[0]},{coord[1]}]" for coord in obj) + "]"
                    input_strs.append(obj_str)
                f.write(",".join(input_strs))
                f.write('],\n')
                
                # Write output
                f.write('        "output": [')
                output_strs = []
                for obj in example["output"]:
                    obj_str = "[" + ",".join(f"[{coord[0]},{coord[1]}]" for coord in obj) + "]"
                    output_strs.append(obj_str)
                f.write(",".join(output_strs))
                f.write(']\n')
                
                if j == len(task_data["train"]) - 1:
                    f.write('      }\n')
                else:
                    f.write('      },\n')
            
            f.write('    ]\n')
            if i == len(task_items) - 1:
                f.write('  }\n')
            else:
                f.write('  },\n')
        
        f.write("}")
    
    print(f"Successfully created {output_file} with {len(result)} tasks")
    return result

if __name__ == "__main__":
    create_pattern_testing_objects()
