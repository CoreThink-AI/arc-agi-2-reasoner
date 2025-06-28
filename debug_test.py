#!/usr/bin/env python3

import asyncio
import sys
import os

# Add the project root to the path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from testing.e2e import get_hints

async def debug_single_task():
    """Debug a single task to understand the pattern structure"""
    task_id = "3e6067c3"
    file_path = f"data/{task_id}.json"
    
    print(f"Debugging task: {task_id}")
    try:
        hint = await get_hints(file_path)
        print(f"Successfully generated hint: {len(hint)} characters")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(debug_single_task())
