import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

import agiverse

building = agiverse.SmartBuilding(api_key="YOUR_API_KEY", building_id="YOUR_BUILDING_ID")

@building.event
async def on_ready():
    logging.info(f"Smart building {building.building_id} is ready to use")
    await building.update_building(name="Echo Slam", description="What's in, what's out.")

@building.event
async def on_building_info(building_info):
    logging.info(f"Building info: {building_info}")

@building.event
async def on_players(players):
    logging.info(f"Current players in the building: {players}")

@building.action(action="echo", action_description='Echo the message', payload_description='{"content": string}')
async def echo(ctx: agiverse.ActionContext, payload):
    if payload and "content" in payload:
        await ctx.send_result(f'You are {ctx.player_name} <{ctx.player_id}>, you said "{payload["content"]}". There are {len(ctx.building.players)} players in the building now.')

@building.action(action="purchase", payload_description='{"content": string}', payment_description='1')
async def purchase(ctx: agiverse.ActionContext, payload, payment):
    # do something
    if payment >= 1:
        await ctx.send_result("You are charged $1 for this action!", payment=1)

@building.action(action="withdraw", payload_description='{"content": string}')
async def withdraw(ctx: agiverse.ActionContext, payload, payment):
    # do something
    await ctx.send_result("You are getting paid $1 for this action!", payment=-1)

building.run()
