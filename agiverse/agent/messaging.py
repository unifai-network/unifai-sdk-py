from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .agent import Agent

import asyncio
import json
import logging
import time
import os
import websockets
from datetime import datetime
from .data import DataTypes
from .utils import (
    distance_obj,
    is_valid_state_data,
    is_valid_map_data,
    is_valid_player_data,
    remove_additional_data,
    seconds_since,
)

logger = logging.getLogger(__name__)

class MessagingHandler:
    agent: "Agent"

    def __init__(self, agent):
        self.agent = agent
        self.message_queue = asyncio.Queue()
        self.map_data = None
        self.nearby_map = None
        self.players_data = None
        self.nearby_players = None
        self.assets_data = None
        self.inventory_data = None
        self.state_data = None
        self.available_actions = None
        self.system_messages = []
        self.other_data = []
        self.last_prompt_time = None
        self.use_model = False
        self.min_model_interval = int(os.getenv('MIN_MODEL_INTERVAL', 5))
        self.max_model_interval = int(os.getenv('MAX_MODEL_INTERVAL', 60))
        self.model_interval = self.min_model_interval
        self.action_id = int(time.time() * 1000)

    async def connect(self, uri):
        self.websocket = await websockets.connect(uri)
        return self

    async def __aenter__(self):
        return self.websocket

    async def __aexit__(self, exc_type, exc, tb):
        await self.websocket.close()

    async def handle_messages(self, websocket):
        receive_task = asyncio.create_task(self._receive_messages(websocket))
        process_task = asyncio.create_task(self._process_messages(websocket))

        done, pending = await asyncio.wait(
            [receive_task, process_task],
            return_when=asyncio.FIRST_COMPLETED,
        )

        for task in done:
            if task.exception():
                raise task.exception()

        for task in pending:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                logger.info(f"Task {task} has been cancelled successfully.")

    async def _receive_messages(self, websocket):
        try:
            async for message in websocket:
                await self.message_queue.put(json.loads(message))
        except websockets.exceptions.ConnectionClosed as e:
            logger.info(f"Receive messages - Connection closed: {e}")
            raise

    async def _process_messages(self, websocket):
        try:
            while True:
                try:
                    message = await asyncio.wait_for(self.message_queue.get(), timeout=60)
                    if not await self._process_message(message):
                        continue
                except asyncio.TimeoutError:
                    self.use_model = True
                    message = None
                except asyncio.CancelledError:
                    raise

                if not self.message_queue.empty():
                    continue

                if len(self.system_messages) == 0:
                    seconds_since_last_prompt = seconds_since(self.last_prompt_time) if self.last_prompt_time else self.max_model_interval
                    if seconds_since_last_prompt < self.model_interval:
                        continue
                    if seconds_since_last_prompt < self.max_model_interval and not self.use_model:
                        continue

                try:
                    response = await self._generate_model_response(websocket)
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    logger.error(f"Error generating model response: {e}")
                    continue

                try:
                    await self._process_model_response(response)
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    logger.error(f"Error processing model response: {e}")
                    continue

                try:
                    await self._send_action(response, websocket)
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    logger.error(f"Error sending action: {e}")
                    raise
        except asyncio.CancelledError:
            logger.info("Process messages has been cancelled.")
            raise
        except Exception as e:
            logger.error(f"Process messages encountered an unexpected error: {e}")
            raise

    async def _process_message(self, message):
        msg_type = message.get('type')
        data = message.get('data')

        if msg_type != 'tickEnd':
            logger.info(f"Received: {msg_type} {data}")

        if msg_type == 'map':
            self.map_data = data
            self.use_model = True
        elif msg_type == 'players':
            self.players_data = data
        elif msg_type == 'assets':
            self.assets_data = data
            self.use_model = True
        elif msg_type == 'inventory':
            self.inventory_data = data
            self.use_model = True
        elif msg_type == 'state':
            self.state_data = data
            if self.state_data['state'] in ['moving', 'learning', 'working']:
                self.model_interval = self.max_model_interval if self.state_data['state'] != 'moving' else self.model_interval
            else:
                self.model_interval = self.min_model_interval
            self.use_model = True
        elif msg_type == 'availableActions':
            self.available_actions = data
            self.use_model = True
        elif msg_type == 'tickEnd':
            pass
        elif msg_type == 'system':
            self.system_messages.append(message)
            self.use_model = True
            self.agent._save_data(DataTypes.SYSTEM_MESSAGE, message)
        else:
            self.other_data.append(message)
            self.use_model = True
            self.agent._save_data(DataTypes.SERVER_MESSAGE, message)

        if is_valid_state_data(self.state_data) and is_valid_map_data(self.map_data, 'buildings'):
            nearby_map = {'buildings': [], 'my_houses': []}
            for building in self.map_data['buildings']:
                building['distance'] = distance_obj(building['entrance'], self.state_data)

            sorted_buildings = sorted(
                self.map_data['buildings'],
                key=lambda b: b.get('distance', float('inf'))
            )

            for building in sorted_buildings:
                remove_additional_data(building)
                distance = building.get('distance', float('inf'))
                if distance > self.agent.vision_range_buildings and len(nearby_map['buildings']) >= self.agent.min_num_buildings:
                    break
                nearby_map['buildings'].append(building)

            if is_valid_map_data(self.assets_data, 'rentedBuildings'):
                for building in self.assets_data['rentedBuildings']:
                    building['distance'] = distance_obj(building['entrance'], self.state_data)

                sorted_rented_buildings = sorted(
                    self.assets_data['rentedBuildings'],
                    key=lambda b: b.get('distance', float('inf'))
                )

                for building in sorted_rented_buildings:
                    remove_additional_data(building)
                    distance = building.get('distance', float('inf'))
                    if distance > self.agent.vision_range_buildings and len(nearby_map['my_houses']) >= self.agent.min_num_buildings:
                        break
                    nearby_map['my_houses'].append(building)

            self.nearby_map = nearby_map

        if is_valid_state_data(self.state_data) and is_valid_player_data(self.players_data):
            nearby_players = []
            for player in self.players_data:
                player['distance'] = distance_obj(player, self.state_data)

            sorted_players = sorted(
                self.players_data,
                key=lambda p: p.get('distance', float('inf'))
            )

            for player in sorted_players:
                if player.get('playerID') == self.state_data.get('playerID'):
                    continue
                distance = player.get('distance', float('inf'))
                if distance > self.agent.vision_range_players and len(nearby_players) >= self.agent.min_num_players:
                    break
                nearby_players.append(player)

            self.nearby_players = nearby_players

        return msg_type == 'tickEnd'

    async def _generate_model_response(self, websocket):
        prompt = self.agent.model_manager.construct_prompt(
            self.agent.get_prompt('agent.agent'),
            character_name=self.agent.name,
            character_info=self.agent.get_prompt('character.info'),
            map_data=self.nearby_map,
            players_data=self.nearby_players,
            assets_data=self.assets_data,
            inventory_data=self.inventory_data,
            state_data=self.state_data,
            available_actions=self.available_actions,
            memory=self.agent.memory,
            long_term_memory=self.agent.long_term_memory,
            planning=self.agent.planning,
            system_messages=self.system_messages,
            messages=self.other_data
        )

        return await self.agent.model_manager.get_model_response(prompt)

    async def _process_model_response(self, response):
        self.agent._save_data(DataTypes.MODEL_RESPONSE, response)

        logger.info('-' * 100)
        logger.info(f"Observation: {response['observation']}")
        logger.info(f"Thought: {response['thought']}")
        logger.info(f"Long term memory: {response.get('longTermMemory', self.agent.long_term_memory)}")
        logger.info(f"Planning: {response.get('planning', self.agent.planning)}")
        logger.info(f"Action: {response['action']}")
        logger.info('-' * 100)

        new_long_term_memory = response.get('longTermMemory', '')
        if new_long_term_memory:
            self.agent.long_term_memory = new_long_term_memory

        new_planning = response.get('planning', '')
        if new_planning:
            self.agent.planning = new_planning

        self.agent.memory.append(response)
        if len(self.agent.memory) > 10:
            self.agent.memory.pop(0)

        self.system_messages = []
        self.other_data = []
        self.last_prompt_time = datetime.now()
        self.use_model = False
        self.model_interval = self.min_model_interval

    async def _send_action(self, response, websocket):
        if not response.get('action') or response['action'].get('action') == 'no action':
            logger.info('No action to perform.')
        else:
            response['action']['actionID'] = self.action_id
            self.action_id += 1
            logger.info(f"Sending action: {response['action']}")
            await websocket.send(json.dumps(response['action']))
