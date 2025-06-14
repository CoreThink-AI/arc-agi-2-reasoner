# import json
# import os
# from visualize import visualize_task, read_json_as_string
# from visualize_objects import  process_grid
#
# train_dir = r"C:\Users\Anugyan\PycharmProjects\ARC-AGI-2-CT\ARC-AGI-2\data\training"
#
# for i in range(1000):
#     task_json_path = os.path.join(train_dir, f"task_{i}.json")
#
#     if not os.path.isfile(task_json_path):
#         continue  # Skip if the file doesn't exist
#
#     task_json_str = read_json_as_string(task_json_path)
#     task_data = json.loads(task_json_str)
#     visualize_task(task_data)
#
#     # Pick one of the train grids (e.g., first one)
#     grid = task_data['train'][1]['input']
#     objects = process_grid(grid)

import json
import os
import csv
from visualize_objects import process_grid
from visualize import visualize_task, read_json_as_string

# Directory where JSON task files are located
train_dir = r"C:\Users\Anugyan\PycharmProjects\ARC-AGI-2-CT\ARC-AGI-2\data\training"

# Output CSV file
feedback_csv_path = "object_detection_feedback.csv"

# Prepare to write feedback
with open(feedback_csv_path, mode='w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["grid_id", "feedback", "num_objects"])

    for i in range(20):
        task_json_path = os.path.join(train_dir, f"task_{i}.json")

        if not os.path.isfile(task_json_path):
            continue  # Skip if the file doesn't exist

        try:
            task_json_str = read_json_as_string(task_json_path)
            task_data = json.loads(task_json_str)
            visualize_task(task_data)
        except Exception as e:
            print(f"Error reading task_{i}: {e}")
            continue

        # Process each train and test grid (input and output)
        for section in ['train', 'test']:
            for idx, pair in enumerate(task_data.get(section, [])):
                for io_type in ['input', 'output']:
                    grid_id = f"task_{i}_{section}_{idx}_{io_type}"
                    print(f"\n--- Processing {grid_id} ---")

                    grid = pair[io_type]
                    objects = process_grid(grid)

                    print(f"Detected {len(objects)} object(s).")
                    feedback = input("Was the object detection correct? (1/0/-1): ").strip().lower()
                    writer.writerow([grid_id, feedback, len(objects)])




