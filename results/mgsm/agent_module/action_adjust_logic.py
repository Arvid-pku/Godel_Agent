

def action_adjust_logic(module_name: str, target_name: str, new_code=str, target_type: str = 'function', operation: str = 'modify'):
    """
    Modify/Add/Delete the source code of the specified logic (function, method, or class) within a given module to 
    improve task-solving ability or create a tool designed specifically to assist in task-solving efficiently.

    Args:
        module_name (str): The name of the module to modify (e.g., 'agent_module').
        target_name (str): The name of the function, method, or class to do operation (e.g., 'solver').
        new_code (str): The new logic as a string (including `def` for functions or `class` or classes). For delete, it can be empty string.
        target_type (str): The type of target ('function', 'class'). Default is 'function'.
        operation (str): The type of operation to perform ('modify', 'add', or 'delete'). Default is 'modify'.

    Raises:
        ValueError: Unknown operation

    Examples:
        >>> modify_logic('agent_module', 'evolve', 'def evolve(agent):\\n    print("New evolve method")', target_type='function')
        >>> modify_logic('agent_module', 'evolve', '', target_type='function', operation='delete')
    """
    if module_name == "agent_module":
        if target_name == "solver":
            if "gpt-4o" in new_code:
                raise ValueError("ONLY model **gpt-3.5-turbo** can be used in solver.")
            if "time.sleep" in new_code:
                raise ValueError("Don't use `time.sleep` in solver.")
        if target_name == "Agent.action_call_llm":
            raise ValueError("Don't modify `action_call_llm`.")
        if target_name == "Agent.action_call_json_format_llm":
            raise ValueError("Don't modify `action_call_json_format_llm`.")

    if "import logging" in new_code or "from logging" in new_code:
        raise ValueError("Don't use `logging`.")

    # Import the module dynamically
    module = importlib.import_module(module_name)
    _target_name = target_name
    print(new_code, end='\n\n')
    # Perform the operation based on type (modify, add, delete)
    if operation in ['modify', 'add']:
        # Compile the new code within the current global and a new local dict
        locals_dict = {}
        exec(compile(new_code, f"running.{module_name}.{target_name}", "exec"), globals(), locals_dict)
        if '.' in target_name:
            class_name, target_name = target_name.split('.')
            if class_name in locals_dict:
                new_target = getattr(locals_dict[class_name], target_name)
                locals_dict.pop(class_name)
            else:
                new_target = locals_dict[target_name]
                locals_dict.pop(target_name)
        else:
            new_target = locals_dict[target_name]
            locals_dict.pop(target_name)
        globals().update(locals_dict)
        
        # Apply the new definition or value to the target
        if '.' in target_name:  # Class attribute
            class_name, target_name = target_name.split('.')
            cls = getattr(module, class_name)
            setattr(cls, target_name, new_target)
            getattr(cls, target_name).__source__ = new_code
            # Add or update the __source__ attribute on the class level to store the full new definition
            if not hasattr(cls, '__source__'):
                cls.__source__ = {}
                for name, method in inspect.getmembers(cls, predicate=inspect.isfunction):
                    cls.__source__[name] = logic.get_source_code(method, name)
            cls.__source__[target_name] = '\n'.join(['    '+code_line for code_line in new_code.split('\n')])

        else:  # Module level attribute
            setattr(module, target_name, new_target)
            getattr(module, target_name).__source__ = new_code

    elif operation == 'delete':
        if '.' in target_name:  # Class attribute
            class_name, target_name = target_name.split('.')
            cls = getattr(module, class_name)
            delattr(cls, target_name)
            if hasattr(cls, '__source__') and target_name in cls.__source__:
                del cls.__source__[target_name]
        else:  # Module level attribute
            delattr(module, target_name)

    else:
        raise ValueError(f"Unknown operation '{operation}'. Expected 'modify', 'add', or 'delete'.")

    return f"Successfully {operation} `{module_name}.{_target_name}`."
