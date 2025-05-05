# unifai-sdk-py

unifai-sdk-py is the Python SDK for Unifai, an AI native platform for dynamic tools and agent to agent communication.

## Installation

```bash
pip install unifai-sdk
```

## Getting your Unifai API key

You can get your API key for free from [Unifai](https://app.unifai.network/).

There are two types of API keys:

- Agent API key: for using toolkits in your own agents.

- Toolkit API key: for creating toolkits that can be used by other agents.

## Using tools

To use tools in your agents, you need an **agent** API key. You can get an agent API key for free at [Unifai](https://app.unifai.network/).

```python
import unifai

tools = unifai.Tools(api_key='xxx')
```

### Tool Types

Unifai provides a flexible system for integrating AI tools in your applications:

#### Dynamic Tools
Dynamic tools are enabled by default, allowing your agent to discover and use tools on-the-fly based on the task at hand.

```python
# Enable dynamic tools (default behavior)
tools_with_dynamic = tools.get_tools(dynamic_tools=True)
```

#### Static Toolkits
Static toolkits allow you to specify entire toolkits to be made available to your agent.

```python
# Get tools from specific toolkits
toolkit_tools = tools.get_tools(
    dynamic_tools=False,  # Optional: disable dynamic tools
    static_toolkits=["toolkit_id_1", "toolkit_id_2"]
)
```

#### Static Actions
Static actions provide granular control, allowing you to specify individual actions (tools).

```python
# Get specific actions
action_tools = tools.get_tools(
    dynamic_tools=False,  # Optional: disable dynamic tools
    static_actionss=["action_id_1", "action_id_2"]
)
```

#### Combining Approaches
You can combine these approaches for a customized tool setup:

```python
# Combine dynamic and static tools
combined_tools = tools.get_tools(
    dynamic_tools=True,
    static_toolkits=["essential_toolkit_id"],
    static_actionss=["critical_action_id"]
)
```

Then you can pass the tools to any OpenAI compatible API. Popular options include:

- OpenAI's native API: For using OpenAI models directly
- [Litellm](https://github.com/BerriAI/litellm): A library that provides a unified OpenAI compatible API to most LLM providers
- [OpenRouter](https://openrouter.ai/docs): A service that gives you access to most LLMs through a single OpenAI compatible API

The tools will work with any API that follows the OpenAI function calling format. This gives you the flexibility to choose the best LLM for your needs while keeping your tools working consistently.

```python
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"content": "Can you tell me what is trending on Google today?", "role": "user"}],
    tools=tools.get_tools(),
)
```

If the response contains tool calls, you can pass them to the tools.call_tools method to get the results. The output will be a list of messages containing the results of the tool calls that can be concatenated to the original messages and passed to the LLM again.

```python
results = await tools.call_tools(response.choices[0].message.tool_calls)
messages.extend(results)
# messages can be passed to the LLM again now
```

Passing the tool calls results back to the LLM might get you more function calls, and you can keep calling the tools until you get a response that doesn't contain any tool calls. For example:

```python
messages = [{"content": "Can you tell me what is trending on Google today?", "role": "user"}]
while True:
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        tools=tools.get_tools(),
    )
    messages.append(response.choices[0].message)
    results = await tools.call_tools(response.choices[0].message.tool_calls)
    if len(results) == 0:
        break
    messages.extend(results)
```

### Using tools in MCP clients

We provide a MCP server to access tools in any [MCP clients](https://modelcontextprotocol.io/clients) such as [Claude Desktop](https://modelcontextprotocol.io/quickstart/user).

The easiest way to run the server is using `uv`, see [Instaling uv](https://docs.astral.sh/uv/getting-started/installation/) if you haven't installed it yet.

Then in your Claude Desktop config:

```json
{
  "mcpServers": {
    "unifai-tools": {
      "command": "uvx",
      "args": [
        "--from",
        "unifai-sdk",
        "unifai-tools-mcp"
      ],
      "env": {
        "UNIFAI_AGENT_API_KEY": ""
      }
    }
  }
}
```

Now your Claude Desktop will be able to access all the tools in Unifai automatically.

## Creating tools

Anyone can create dynamic tools in Unifai by creating a toolkit.

A toolkit is a collection of tools that are connected to the Unifai infrastructure, and can be searched and used by agents dynamically.

Initialize a toolkit client with your **toolkit** API key. You can get a toolkit API key for free at [Unifai](https://app.unifai.network/).

```python
import unifai

toolkit = unifai.Toolkit(api_key='xxx')
```

Update the toolkit name and/or description if you need:

```python
await toolkit.update_toolkit(name="Echo Slam", description="What's in, what's out.")
```

or running it as a synchronous method with asyncio.run():

```python
asyncio.run(toolkit.update_toolkit(name="Echo Slam", description="What's in, what's out."))
```

Register action handlers:

```python
@toolkit.action(
    action="echo",
    action_description='Echo the message',
    payload_description={"content": {"type": "string"}},
)
def echo(ctx: unifai.ActionContext, payload={}): # can be an async function too
    return ctx.Result(f'You are agent <{ctx.agent_id}>, you said "{payload.get("content")}".')
```

Note that `payload_description` can be any string or a dict that contains enough information for agents to understand the payload format. It doesn't have to be in certain format, as long as agents can understand it as nautural language and generate correct payload. Think of it as the comments and docs for your API, agents read it and decide what parameters to use.

Start the toolkit:

```python
await toolkit.run()
```

or running it as a synchronous method with asyncio.run():

```python
asyncio.run(toolkit.run())
```

## Examples

You can find examples in the `examples` directory.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.
