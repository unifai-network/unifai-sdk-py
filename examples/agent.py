import agiverse

agent = agiverse.Agent(api_key="Your API Key", name="Your character name")

character_info = agent.get_prompt("character.info")
character_info += "\nYou are a resourceful person who values both success and helping others."
agent.set_prompt("character.info", character_info)

agent.run()
