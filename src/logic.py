import os
import io
import gc
import re
import ast
import sys
import json
import typing
import shutil
import openai
import inspect
import functools
import traceback
import importlib
import subprocess
import contextlib
import numpy as np
import concurrent.futures

def get_source_code(obj, obj_name):
    if hasattr(obj, '__source__'):
        if inspect.isclass(obj):
            return f'class {obj_name}:\n' + "\n\n".join(f"{code}" for code in obj.__source__.values())
        return obj.__source__
    else:
        return inspect.getsource(obj)
    
def merge_and_clean(dump_folder='../dumped_agent'):
    """
    Merges Python files in specific subfolders, copies certain files to the root of the dump folder,
    and performs clean-up by removing folders after merging.
    """
    
    def merge_py_files(source_folder, exclude_files, output_file):
        """
        Merges Python files in a folder into one output file, excluding specified files.
        """
        py_files = [f for f in os.listdir(source_folder) if f.endswith('.py') and f not in exclude_files]
        merged_code = ""
        for py_file in py_files:
            with open(os.path.join(source_folder, py_file), 'r') as f:
                content = f.read()
                merged_code += f"# {py_file}\n" + content + "\n\n"
        with open(os.path.join(dump_folder, output_file), 'w') as f:
            f.write(merged_code)
        print(f"Merged {len(py_files)} files into {output_file}.")
    
    def copy_and_merge_imports(existing_file, merged_file):
        """
        Copies import lines from the existing file to the top of the merged file.
        """
        if not os.path.exists(existing_file):
            return
        
        # Read the existing imports from the original file
        with open(existing_file, 'r') as f:
            lines = f.readlines()
        imports = [line for line in lines if line.startswith("import ") or line.startswith("from ")]
        
        # Prepend the imports to the merged file
        with open(os.path.join(dump_folder, merged_file), 'r') as f:
            merged_code = f.read()
        
        with open(os.path.join(dump_folder, merged_file), 'w') as f:
            f.write("".join(imports) + "\n" + merged_code)
        print(f"Copied imports from {existing_file} to {merged_file}.")
    
    # Merge Python files in agent_module and task folders
    agent_module_folder = os.path.join(dump_folder, 'agent_module')
    task_folder = os.path.join(dump_folder, 'task')
    
    if os.path.exists(agent_module_folder):
        merge_py_files(agent_module_folder, exclude_files=[
            'ThreadPoolExecutor.py', 'Any.py', 'partial.py', 'as_completed.py', 'lru_cache.py', 'Game24Task.py', 'MathTask.py'
        ], output_file='agent_module.py')
    
    if os.path.exists(task_folder):
        merge_py_files(task_folder, exclude_files=[], output_file='task.py')
    
    # Copy key.env, goal_prompt.md, and main.py to the dump_folder
    for file_name in ['key.env', 'goal_prompt.md', 'main.py']:
        source_file = os.path.join(os.getcwd(), file_name)
        if os.path.exists(source_file):
            shutil.copy(source_file, dump_folder)
            print(f"Copied {file_name} to {dump_folder}.")
    
    # Copy imports from the original agent_module.py and task.py to the new merged files
    current_folder = os.getcwd()
    copy_and_merge_imports(os.path.join(current_folder, 'agent_module.py'), 'agent_module.py')
    copy_and_merge_imports(os.path.join(current_folder, 'task.py'), 'task.py')
    
    # Clean up: remove the agent_module and task folders
    if os.path.exists(agent_module_folder):
        shutil.rmtree(agent_module_folder)
        print(f"Removed folder {agent_module_folder}.")
    
    if os.path.exists(task_folder):
        shutil.rmtree(task_folder)
        print(f"Removed folder {task_folder}.")
    
    print(f"Merge and clean-up process completed.")

def store_all_logic(dump_folder='../dumped_agent'):
    """
    Dumps all custom logic (functions, methods, classes) from memory into new files in a specified folder.
    Adds necessary imports and post-processes the files to make them runnable.
    """
    # Create the dump folder if it doesn't exist, or clean it up if it exists
    if os.path.exists(dump_folder):
        shutil.rmtree(dump_folder)  # Remove the old folder and create a fresh one
    os.makedirs(dump_folder)

    project_directory = os.getcwd()

    def get_imports(source_code):
        """
        Analyze the source code to extract any missing imports.
        This is a basic implementation. For more complex imports, you can analyze dependencies.
        """
        tree = ast.parse(source_code)
        imports = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))]
        import_lines = [ast.unparse(node) for node in imports]  # Get the actual import lines
        return "\n".join(import_lines) + "\n"

    def post_process(source_code):
        """
        Add missing imports or dependencies to make the dumped code runnable.
        """
        imports = get_imports(source_code)
        return f"{imports}\n{source_code}"

    def dump_object(obj, name, folder):
        try:
            source_code = get_source_code(obj, name)  # Assume get_custom_sources() retrieves the function/class source
            source_code = post_process(source_code)      # Post-process the code to add imports/dependencies
            file_name = f"{name}.py"
            file_path = os.path.join(folder, file_name)
            with open(file_path, 'w') as file:
                file.write(source_code)
            return f"Dumped {name} to {file_path}."
        except Exception as e:
            return f"Failed to dump {name}: {str(e)}"

    def process_module(module_name, module):
        if not hasattr(module, '__file__'):
            return []

        try:
            module_file_path = os.path.abspath(inspect.getfile(module))
        except Exception:
            return []

        if not module_file_path.startswith(project_directory):
            return []

        module_folder = os.path.join(dump_folder, module_name)
        os.makedirs(module_folder, exist_ok=True)

        tasks = []
        for name, obj in vars(module).items():
            if inspect.isfunction(obj) or inspect.isclass(obj):
                tasks.append((obj, name, module_folder))
        return tasks

    all_tasks = []
    for module_name, module in sys.modules.items():
        all_tasks.extend(process_module(module_name, module))

    # Process global objects
    global_folder = os.path.join(dump_folder, "global_objects")
    os.makedirs(global_folder, exist_ok=True)
    for name, obj in globals().items():
        if inspect.isfunction(obj) or inspect.isclass(obj):
            all_tasks.append((obj, name, global_folder))

    # Use ThreadPoolExecutor for parallel processing
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(dump_object, *task) for task in all_tasks]
        for future in concurrent.futures.as_completed(futures):
            print(future.result())
    # merge_and_clean(dump_folder)
    print(f"All logic dumped into folder: {dump_folder}")