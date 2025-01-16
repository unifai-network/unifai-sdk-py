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
async def echo(ctx: unifai.ActionContext, payload, payment):
    if payload and "content" in payload:
        await ctx.send_result(f'You are agent <{ctx.agent_id}>, you said "{payload["content"]}".')

asyncio.run(toolkit.run())
