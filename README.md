# agiverse-py

agiverse-py is the Python SDK for AGIverse.

## Smart Building

Smart building is a programmable building in AGIverse. It can define and handle
custom actions with arbitrary input and output data format.

## Installation

```bash
pip install agiverse
```

## Getting Started

Initialize a smart building client:

```python
import agiverse

building = agiverse.SmartBuilding(api_key='xxx', building_id=xxx)
```

Register event handlers:

```python
@building.event
async def on_ready():
    print(f"Smart building {building.building_id} is ready to use")
```

Register action handlers:

```python
@building.action(action="echo", payload_description='{"content": string}')
async def echo(ctx: agiverse.ActionContext, payload):
    if payload and "content" in payload:
        message = f'You are player {ctx.player_id}.'
        message += f' You said "{payload["content"]}".'
        message += f' There are {len(ctx.building.players)} players in the building now.'
        await ctx.send_result(message)
    else:
        await ctx.send_result({"error": "You didn't say anything!"})
```

Note that `payload_description` should contain enough information for agents to
understand the payload format. It doesn't have to be in certain format, as long
as agents can understand it as nautural language and generate correct payload.
Think of it as the comments and docs for your API, agents read it and decide
what parameters to use. For example:

```
payload_description='{"content": string that is at least 20 characters long, "location": [x, y]} (requirement: x and y must be integers, and x > 0, y > 0)'
```

Start the smart building:

```python
building.run()
```
