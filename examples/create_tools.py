import asyncio
import unifai

toolkit = unifai.Toolkit(api_key="YOUR_TOOLKIT_API_KEY")

asyncio.run(toolkit.update_toolkit(name="Echo Slam", description="What's in, what's out."))

@toolkit.event
async def on_ready():
    print(f"Toolkit is ready to use")

@toolkit.action(
    action="echo",
    action_description='Echo the message',
    payload_description={"content": {"type": "string"}},
)
def echo(ctx: unifai.ActionContext, payload={}) -> unifai.ActionResult: # can be an async function too
    return ctx.Result(f'You are agent <{ctx.agent_id}>, you said "{payload.get("content")}".')

asyncio.run(toolkit.run())
