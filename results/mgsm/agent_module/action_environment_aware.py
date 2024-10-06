

def action_environment_aware(agent: AgentBase):
    """
    Reflect and summarize available resources of the current runtime environment including variables, functions, modules, and external libraries.

    Returns:
        summary (str): Summary of available resources.
    """
    def summarize_items(items, header):
        summary = [header]
        for name, value in items:
            if not name.startswith('__'):
                if name in ['goal_prompt']:
                    summary.append(f"- {name} = Your {name}.")
                elif name in ['optimize_history', 'function_map', 'action_functions']:
                    summary.append(f"- {name} = The length of your {name} is {len(getattr(agent, name))}.")
                elif name in ['logic']:
                    pass
                else:
                    summary.append(f"- {name} = {value}")
        if len(summary) == 1:
            summary.append("- None")
        return summary
    
    summary = []
    
    global_vars = [(k, v) for k, v in globals().items() if not k.startswith('__') and k != "AgentBase"]
    functions = [(k, v) for k, v in global_vars if inspect.isfunction(v)]
    calsses = [(k, v) for k, v in global_vars if inspect.isclass(v)]
    modules = [(k, v) for k, v in global_vars if inspect.ismodule(v)]
    variables = [(k, v) for k, v in global_vars if not (inspect.isfunction(v) or inspect.isclass(v) or inspect.ismodule(v))]
    
    summary.extend(summarize_items(functions, "\nGlobal Functions:"))
    summary.extend(summarize_items(modules, "\nGlobal Modules:"))
    summary.extend(summarize_items(variables, "\nGlobal Variables:"))
    summary.extend(summarize_items(calsses, "\nGlobal Calsses:"))
    
    methods = inspect.getmembers(agent, inspect.ismethod)
    attributes = inspect.getmembers(agent, lambda x: not inspect.ismethod(x))
    
    summary.extend(summarize_items(methods, "\nCurrent Agent Instance's Methods:"))
    summary.extend(summarize_items(attributes, "\nCurrent Agent Instance's Attributes:"))

    return "\n".join(summary).strip()
