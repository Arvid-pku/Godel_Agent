from agent_module import Agent, self_evolving_agent

if __name__ == "__main__":
    key_path = "src/key.env"
    for _ in range(1):
        self_evolving_agent = Agent(goal_prompt_path="src/goal_prompt.md", key_path=key_path)
        self_evolving_agent.reinit()
        self_evolving_agent.evolve()