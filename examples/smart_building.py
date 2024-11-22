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

@building.event
async def on_building_info(building_info):
    logging.info(f"Building info: {building_info}")

@building.event
async def on_players(players):
    logging.info(f"Current players in the building: {players}")

@building.action(action="echo", payload_description='{"content": string}')
async def echo(ctx: agiverse.ActionContext, payload):
    if payload and "content" in payload:
        await ctx.send_result(f'You are player {ctx.player_id}, you said "{payload["content"]}". There are {len(ctx.building.players)} players in the building now.')
    else:
        await ctx.send_result({"error": "You didn't say anything!"})

building.run()
