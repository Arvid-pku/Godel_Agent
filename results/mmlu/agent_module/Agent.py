

class Agent(AgentBase):
    def __init__(agent, api_key=None, goal_prompt_path='goal_prompt.md', key_path='key.env'):
        # Load configurations
        agent.goal_prompt = open(goal_prompt_path, 'r').read()
        agent.goal_task = mmlu.MMLU_Task()
        if api_key is None:
            api_key = open(key_path, 'r').read().strip()
        openai.api_key = api_key
        agent.client = openai.OpenAI(api_key=api_key)

        # Initialize optimization history and iterations

        agent.action_functions = [
            {
                "type": "function",
                "function": {
                    "name": "action_display_analysis",
                    "description": "Display an analysis of the current state, including available resources, logic (of solver or other actions) and evaluation feedbacks from the target task, and reasons or plans for the next actions based on this analysis.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "analysis": {
                                "type": "string",
                                "description": "A detailed analysis of the current state, including reasons or plans for the following actions."
                            }
                        },
                        "required": ["analysis"],
                        "additionalProperties": False,
                    },
                    "strict": True
                }
            },
            # {
            #     "type": "function",
            #     "function": {
            #         "name": "action_environment_aware",
            #         "description": "Reflect and summarize available resources of the current runtime environment including variables, functions, modules, and external libraries.",
            #         "parameters": {
            #             "type": "object",
            #             "properties": {},
            #             "required": [],
            #             "additionalProperties": False
            #         },
            #         "strict": True
            #     }
            # },
            {
                "type": "function",
                "function": {
                    "name": "action_read_logic",
                    "description": "Reads the source code of the specified logic (function, method, or class) within a given module.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "module_name": {
                                "type": "string",
                                "description": "The module where the logic resides."
                            },
                            "target_name": {
                                "type": "string",
                                "description": "The name of the function, method, or class to read. If the target_name contains a dot, it refers to a method within a class (e.g., 'Agent.action_call_llm')."
                            }
                        },
                        "required": ["module_name", "target_name"],
                        "additionalProperties": False
                    },
                    "strict": True
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "action_adjust_logic",
                    "description": "Modify/Add/Delete the source code of the specified logic (function, method, or class) within a given module to improve task-solving ability or create a tool designed specifically to assist in task-solving efficiently.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "module_name": {
                                "type": "string",
                                "description": "The module where the logic resides."
                            },
                            "target_name": {
                                "type": "string",
                                "description": "The name of the function, method, or class to modify/add/delete. If the target_name contains a dot, it refers to a method within a class."
                            },
                            "new_code": {
                                "type": "string",
                                "description": "The new logic as a string. (Ensure there is no extra indentation in new_code)"
                            },
                            "target_type": {
                                "type": "string",
                                "enum": ["function", "class"],
                                "description": "The type of target."
                            },
                            "operation": {
                                "type": "string",
                                "enum": ["modify", "add", "delete"],
                                "description": "The operation to perform."
                            }
                        },
                        "required": ["module_name", "target_name", "new_code", "target_type", "operation"],
                        "additionalProperties": False
                    },
                    "strict": True
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "action_run_code",
                    "description": "Execute Python or shell code and capture the output, errors, and return value. (Running python code can get and store objects designed specifically to assist in task-solving efficiently, such as prompts)",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "code_type": {
                                "type": "string",
                                "enum": ["python", "bash"],
                                "description": "The type of code to execute."
                            },
                            "code": {
                                "type": "string",
                                "description": "The code to execute as a string."
                            }
                        },
                        "required": ["code_type", "code"],
                        "additionalProperties": False
                    },
                    "strict": True
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "action_call_json_format_llm",
                    "description": "Call an external LLM for assistance with gathering insights, refining strategies, correcting errors, and solving complex problems. Output response in JSON format.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "model": {
                                "enum": ["gpt-4o-mini", "gpt-4o"],
                                "description": "ID of the model to use."
                            },
                            "messages": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "role": {
                                            "enum": ["system", "assistant", "user"]
                                        },
                                        "content": {"type": "string"}
                                    },
                                    "required": ["role", "content"],
                                    "additionalProperties": False
                                },
                                "description": "A list of messages comprising the conversation so far."
                            },
                            "temperature": {
                                "type": "number",
                                "description": "What sampling temperature to use. Higher values will make the output more random, while lower values will make it more focused and deterministic."
                            },
                            "role": {
                                "type": "string",
                                "description": "The role that LLM play."
                            },
                            "return_dict_keys": {
                                "type": "array",
                                "items": {
                                    "type": "string"
                                },
                                "description": "An array containing the names of the keys that should be present in the returned dictionary."
                            },
                            "requirements": {
                                "type": "string",
                                "description": "A string that specifies the conditions required to perform a call to the LLM."
                            }
                        },
                        "required": ["model", "messages", "temperature", "role", "return_dict_keys", "requirements"],
                        "additionalProperties": False
                    },
                    "strict": True
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "action_evaluate_on_task",
                    "description": "Evaluate the current solver on the goal task samples and return the evaluation feedback including valid set accuracy, test set accuray, test sample inputs, model outputs and valid sample answer.",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": [],
                        "additionalProperties": False
                    },
                    "strict": True
                }
            }
        ]

        agent.optimize_history = []

    def reinit(agent):
        agent.optimize_history = []
        first_aware_content = action_environment_aware(agent)
        solver_logic = action_read_logic("agent_module", "solver")
        
        print(first_aware_content, end="\n\n")
        print(solver_logic, end="\n\n")

        # agent.optimize_history.append({"role": "user", "content": first_aware_content})
        agent.optimize_history.append({"role": "user", "content": "The logic of solver:\n" + solver_logic})

    def execute_action(agent, actions: typing.Dict):
        """
        Executes the function called by the model and returns the result.
        """
        
        is_reinit = False
        for tool_call in actions['tool_calls']:
            print("tool call:", tool_call, end="\n\n")
            try:
                action_counter[tool_call['function']['name']] += 1
                arguments = json.loads(tool_call['function']['arguments']) if tool_call['function']['arguments'] else {}
                if tool_call['function']['name'] == "action_display_analysis":
                    result = action_display_analysis(**arguments)

                elif tool_call['function']['name'] == "action_environment_aware":
                    result = action_environment_aware(agent, **arguments)

                elif tool_call['function']['name'] == "action_read_logic":
                    result = action_read_logic(**arguments)

                elif tool_call['function']['name'] == "action_adjust_logic":
                    result = action_adjust_logic(**arguments)

                elif tool_call['function']['name'] == "action_run_code":
                    result = action_run_code(**arguments)
                    if arguments.get("code_type", None) == "python" and "self_evolving_agent.reinit()" in arguments.get("code", ""):
                        is_reinit = True
                elif tool_call['function']['name'] == "action_call_llm":
                    result = agent.action_call_llm(**arguments)
                    print(result[0])

                elif tool_call['function']['name'] == 'action_call_json_format_llm':
                    result = agent.action_call_json_format_llm(**arguments)
                    try:
                        print(json.loads(result[0]))
                    except:
                        print(result[0])

                elif tool_call['function']['name'] == "action_evaluate_on_task":
                    result = action_evaluate_on_task(agent.goal_task, functools.partial(solver, agent))
                else:
                    raise ValueError(f"Unknown function name: {tool_call['function']['name']}")

            except Exception as e:
                action_counter["error_handle"] += 1
                exception_stringio = io.StringIO()
                traceback.print_exc(file=exception_stringio)
                result = "Error " + exception_stringio.getvalue()
                exception_stringio.close()

            print("tool call result:\n", result, sep="", end="\n\n")
            if is_reinit:
                break
            agent.optimize_history.append({"role": "tool", 
                                           "content": result, 
                                            "tool_call_id": tool_call['id']})


        print("Action Counter:", action_counter, end='\n\n')
        if action_counter["evolve"] >= 100:
            sys.exit(1)
        print("Agent Evolve", end="\n\n")
        
        agent.evolve()
        # try:
        #     agent.evolve()
        # except Exception as e:
        #     exception_stringio = io.StringIO()
        #     traceback.print_exc(file=exception_stringio, limit=3)
        #     result = exception_stringio.getvalue()
        #     exception_stringio.close()
        #     print("evolve error result:\n", result, sep="", end="\n\n")
        #     agent.optimize_history.append({"role": "user", "content": f'There are some errors in your code:\nError {result}.\nTherefore, the action has been aborted.'})
        #     agent.evolve = stored_evolve_f
        #     agent.evolve()

    def evolve(agent):
        """
        Evolves the agent by prompting the LLM to suggest improvements.
        """
        print('-' * 120)
        action_counter["evolve"] += 1

        tool_call_ids = set()
        remain_optimize_history = []
        for message in agent.optimize_history[-10:]:
            if message["role"] == "assistant" and message["tool_calls"]:
                tool_call_ids = set()
                for tool_call in message["tool_calls"]:
                    tool_call_ids.add(tool_call["id"])
            if message["role"] == "tool" and message["tool_call_id"] not in tool_call_ids:
                print(f"pop item: {message}", end='\n\n')
                continue
            remain_optimize_history.append(message)
        agent.optimize_history = remain_optimize_history

        messages = [{"role": "system", "name": "Principles", "content": agent.goal_prompt}, 
                    {"role": "system", "name": "Environment", "content": action_environment_aware(agent)},
                    *agent.optimize_history]
        try:
            response = agent.action_call_llm(messages=messages, model="gpt-4o", response_format="text", tools=agent.action_functions, tool_choice="required")
        except Exception as e:
            print(repr(e))
            for message in messages:
                print(message)
            sys.exit(1)
        
        agent.optimize_history.append(response[0])
        agent.execute_action(response[0])

    def action_call_json_format_llm(
        agent,
        *,
        messages: typing.List[typing.Dict[str, str]], 
        model: typing.Literal["gpt-3.5-turbo", "gpt-4o-mini", "gpt-4o"] = "gpt-4o-mini", 
        temperature: float = 1.0, 
        max_completion_tokens: int = 4096, 
        num_of_response: int = 1,
        role: str = "task solver", 
        return_dict_keys: typing.List[str] = [], 
        requirements: str = "", 
    ):
        system_prompt = (
            f"You are a helpful {role}.\n"
            f"Reply in JSON format, ONLY using the keys {return_dict_keys}.\n"
            f"Requirements:\n{requirements}"
        ).strip()
        _messages = [{"role": "system", "content": system_prompt}, *messages]
        return_dicts = agent.action_call_llm(model=model,
                                    messages=_messages, 
                                    temperature=temperature,
                                    max_completion_tokens=max_completion_tokens,
                                    n=num_of_response,
                                    response_format="json")
        
        for key in return_dict_keys:
            for return_dict in return_dicts:
                if key not in return_dict:
                    return_dict[key] = f"NO {key} IN DICTIONARY"
        return return_dicts
    
    def action_call_llm(
        agent, 
        *,
        model: typing.Literal["gpt-3.5-turbo", "gpt-4o-mini", "gpt-4o"] = "gpt-4o-mini", 
        messages: typing.List[typing.Dict[str, str]], 
        temperature: float = 1.0, 
        max_completion_tokens: int = 4096, 
        n: int = 1,
        response_format: typing.Literal["text", "json", "json_object"] = "text", 
        tools=None, 
        tool_choice=None,
    ):
        """
        Sends a request to the OpenAI LLM with a system prompt and user message, and returns the response.

        Args:
            agent (Agent): The OpenAI client instance used to interact with the LLM.
            messages (List[Dict[str, str]]): A list of message dictionaries (conversation history).
            response_format (str): The desired format of the LLM's output.
            model (str): Specifies which LLM model to use.
            temperature (float): A float value controlling the randomness of the model's responses. Higher values (e.g., 1.0) increase creativity, while lower values (e.g., 0.1) make the responses more focused and deterministic.
            max_completion_tokens: An integer defining the maximum number of tokens in the completion response, up to 4096.
            n (int): The number of chat completion choices to generate for each input message.

        Returns:
            response (dict): The response from the OpenAI LLM.
        """
        try:
            if response_format == "json":
                response_format = "json_object"
            
            import copy
            messages = copy.deepcopy(messages)
            for message in messages:
                message["content"] = str(message["content"])
            
            kwargs = {
                "n": n,
                "model": model,
                "messages": messages,
                "response_format": {"type": response_format if response_format == "json_object" else "text"}, 
                "temperature": temperature,
                "max_completion_tokens": max_completion_tokens
            }

            if tools is not None:
                kwargs["tools"] = tools
                kwargs["tool_choice"] = tool_choice

            response = agent.client.chat.completions.create(**kwargs).to_dict() # to Python dictionary
            
            def try_parse_json(content):
                try:
                    return json.loads(content)
                except:
                    return {"JSONDecodeError": content}

            if response_format == "text":
                return [choice["message"] for choice in response["choices"]]
            else:
                return [try_parse_json(choice["message"]["content"]) for choice in response["choices"]]
        except Exception as e:
            raise e
