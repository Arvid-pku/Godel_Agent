

def action_run_code(code_type: str, code: str, timeout: float = 30.0) -> str:
    """
    Execute Python or shell code and capture the output, errors, and return value. 
    (Running python code can get and store objects designed specifically to assist in task-solving efficiently, such as prompts)
    
    Args:
        code_type (str): The type of code to execute ('python' or 'bash').
        code (str): The code to execute as a string.
        timeout (float): Maximum execution time in seconds (default: 30.0).
    
    Returns:
        result_str (str): A string summarizing the output, errors, and return value.
    """
    
    def safe_eval(expr: str, globals_dict, locals_dict):
        """Safely evaluate an expression."""
        try:
            tree = ast.parse(expr, mode='eval')
            return eval(compile(tree, '<string>', 'eval'), globals_dict, locals_dict)
        except Exception:
            return None

    if code_type.lower() == 'python':
        output = io.StringIO()
        error_output = io.StringIO()
        return_value = None
        
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(error_output):                
            locals_dict = {}
            exec(code, globals(), locals_dict)
            globals().update(locals_dict)
            
            # Safely evaluate the last expression
            return_value = safe_eval(code.splitlines()[-1], globals(), locals_dict)
        
        result = {
            "output": output.getvalue(),
            "errors": error_output.getvalue(),
            "return_value": return_value
        }
        
        output.close()
        error_output.close()
    
    elif code_type.lower() == 'bash':
        try:
            process = subprocess.run(
                code,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            result = {
                "output": process.stdout,
                "errors": process.stderr,
                "return_value": process.returncode
            }
        except subprocess.TimeoutExpired:
            result = {
                "output": "",
                "errors": f"Command timed out after {timeout} seconds",
                "return_value": None
            }
        except Exception as e:
            result = {
                "output": "",
                "errors": repr(e),
                "return_value": None
            }
            
    else:
        return "Error: Unsupported code_type. Only 'python' and 'bash' are supported."

    # Format the result
    result_str = f"Execution Summary ({code_type.capitalize()}):\n"
    if result["output"]:
        result_str += f"Output:\n{result['output']}\n"
    if result["errors"]:
        result_str += f"Errors:\n{result['errors']}\n"
    if result["return_value"] is not None:
        result_str += f"Return Value: {result['return_value']}\n"
    
    return result_str or "No output, errors, or return value."
