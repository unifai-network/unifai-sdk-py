import asyncio
import inspect
import json
import logging
from websockets import connect, ConnectionClosedError
from .context import ActionContext
from ..common.const import DEFAULT_WS_ENDPOINT

logger = logging.getLogger(__name__)

class SmartBuilding:
    def __init__(self, api_key, building_id, reconnect_interval=5):
        """
        A smart building in AGIverse.

        :param api_key: Your API key.
        :param building_id: The building ID to connect to.
        :param reconnect_interval: Time in seconds between reconnection attempts.
        """
        self.api_key = api_key
        self.building_id = building_id
        self._event_handlers = {}
        self._action_handlers = {}
        self.players = []
        self.building_info = {}
        self._ws = None
        self._reconnect_interval = reconnect_interval
        self._reconnect = True
        self.set_ws_endpoint(DEFAULT_WS_ENDPOINT)

    def set_ws_endpoint(self, endpoint):
        self.ws_uri = f"{endpoint}?type=building&api-key={self.api_key}&building-id={self.building_id}"

    def event(self, func):
        """
        Decorator to register an event handler.
        """
        self._event_handlers[func.__name__] = func
        return func

    def action(self, action, action_description='', payload_description='', payment_description=''):
        """
        Decorator to register an action handler.
        """
        def decorator(func):
            self._action_handlers[action] = {
                'func': func,
                'action_description': action_description,
                'payload_description': payload_description,
                'payment_description': payment_description
            }
            return func
        return decorator

    async def update_building(self, name = None, description = None):
        data = {
            "buildingID": self.building_id,
        }
        if name:
            data["name"] = name
        if description:
            data["description"] = description
        await self._ws.send(json.dumps({
            "type": "updateBuilding",
            "data": data,
        }))

    async def _handle_messages(self):
        while True:
            try:
                message = await self._ws.recv()
                logger.debug(f"Received raw message: {message}")
                try:
                    msg = json.loads(message)
                except json.JSONDecodeError:
                    logger.warning("Received a non-JSON message.")
                    continue
                if not isinstance(msg, dict):
                    logger.warning("Received message is not a dictionary")
                    continue
                msg_type = msg.get("type")
                if msg_type == "building":
                    if not isinstance(msg.get("data"), dict):
                        logger.warning("Received building message with non-dictionary data")
                        continue
                    self.building_info = msg["data"]
                    if 'on_building_info' in self._event_handlers:
                        await self._event_handlers['on_building_info'](self.building_info)
                elif msg_type == "players":
                    if not isinstance(msg.get("data"), list):
                        logger.warning("Received players message with non-list data")
                        continue
                    self.players = msg["data"]
                    if 'on_players' in self._event_handlers:
                        await self._event_handlers['on_players'](self.players)
                elif msg_type == "action":
                    if not isinstance(msg.get("data"), dict):
                        logger.warning("Received action message with non-dictionary data")
                        continue
                    action_name = msg["data"].get("action")
                    action_handler = self._action_handlers.get(action_name)
                    if action_handler:
                        ctx = ActionContext(
                            player_id=msg["data"].get("playerID"),
                            player_name=msg["data"].get("playerName"),
                            building=self,
                            websocket=self._ws,
                            action_id=msg["data"].get("actionID"),
                            action_name=action_name
                        )
                        payload = msg["data"].get("payload")
                        payment = msg["data"].get("payment")

                        num_params = len(inspect.signature(action_handler['func']).parameters)

                        if num_params == 1:
                            await action_handler['func'](ctx)
                        elif num_params == 2:
                            await action_handler['func'](ctx, payload)
                        elif num_params == 3:
                            await action_handler['func'](ctx, payload, payment)
                        else:
                            logger.warning(f"Handler for action '{action_name}' has an unexpected number of parameters")
                    else:
                        logger.warning(f"No handler for action '{action_name}'")
            except ConnectionClosedError:
                logger.warning("Connection closed, attempting to reconnect...")
                break
            except Exception as e:
                logger.error(f"An error occurred: {e}")
                break

    async def _connect(self):
        while self._reconnect:
            try:
                async with connect(self.ws_uri) as ws:
                    self._ws = ws
                    logger.info("WebSocket connection established.")

                    set_actions_message = {
                        "type": "registerActions",
                        "data": {
                            "actions": {
                                action: {
                                    'description': handler['action_description'],
                                    'payload': handler['payload_description'],
                                    'payment': handler['payment_description']
                                }
                                for action, handler in self._action_handlers.items()
                            }
                        }
                    }
                    await self._ws.send(json.dumps(set_actions_message))

                    if 'on_ready' in self._event_handlers:
                        await self._event_handlers['on_ready']()
                    await self._handle_messages()
            except Exception as e:
                logger.error(f"An error occurred: {e}")
                logger.info(f"Reconnecting in {self._reconnect_interval} seconds...")
                await asyncio.sleep(self._reconnect_interval)

    def run(self):
        """
        Starts the connection to the smart building server.
        """
        asyncio.run(self._connect())
