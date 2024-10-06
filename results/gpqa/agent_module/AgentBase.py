

class AgentBase:
    def execute_action(agent, *args, **kwargs):
        raise NotImplementedError("execute_action hasn't been implemented")
    
    def action_call_llm(agent, *args, **kwargs):
        raise NotImplementedError("action_call_llm hasn't been implemented")
