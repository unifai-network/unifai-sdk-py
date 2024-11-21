import asyncio
import json
import logging
import sys
from websockets import connect, ConnectionClosedError
from .context import ActionContext

logging.basicConfig(level=logging.WARNING, stream=sys.stdout)

class SmartBuilding:
    def __init__(self, api_key, building_id, reconnect_interval=5):
        """
        A smart building in AGIverse.

        :param api_key: Your API key.
        :param building_id: The building ID to connect to.
        :param reconnect_interval: Time in seconds between reconnection attempts.
        :param endpoint: The WebSocket endpoint URL to connect to.
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
        self.set_endpoint("wss://backend.agiverse.io/ws")

    def set_endpoint(self, endpoint):
        self.uri = f"{endpoint}?type=building&api-key={self.api_key}&building-id={self.building_id}"

    def event(self, func):
        """
        Decorator to register an event handler.
        """
        self._event_handlers[func.__name__] = func
        return func

    def action(self, action, payload_description=''):
        """
        Decorator to register an action handler.
        """
        def decorator(func):
            self._action_handlers[action] = {
                'func': func,
                'payload_description': payload_description
            }
            return func
        return decorator

    async def _handle_messages(self):
        while True:
            try:
                message = await self._ws.recv()
                logging.debug(f"Received raw message: {message}")
                try:
                    msg = json.loads(message)
                except json.JSONDecodeError:
                    logging.warning("Received a non-JSON message.")
                    continue
                if not isinstance(msg, dict):
                    logging.warning("Received message is not a dictionary")
                    continue
                msg_type = msg.get("type")
                if msg_type == "building":
                    if not isinstance(msg.get("data"), dict):
                        logging.warning("Received building message with non-dictionary data")
                        continue
                    self.building_info = msg["data"]
                    if 'on_building_info' in self._event_handlers:
                        await self._event_handlers['on_building_info'](self.building_info)
                elif msg_type == "players":
                    if not isinstance(msg.get("data"), list):
                        logging.warning("Received players message with non-list data")
                        continue
                    self.players = msg["data"]
                    if 'on_players' in self._event_handlers:
                        await self._event_handlers['on_players'](self.players)
                elif msg_type == "action":
                    if not isinstance(msg.get("data"), dict):
                        logging.warning("Received action message with non-dictionary data")
                        continue
                    action_name = msg["data"].get("action")
                    action_handler = self._action_handlers.get(action_name)
                    if action_handler:
                        ctx = ActionContext(
                            player_id=msg["data"].get("playerID"),
                            building=self,
                            websocket=self._ws,
                            action_id=msg["data"].get("actionID"),
                            action_name=action_name
                        )
                        payload = msg["data"].get("payload")
                        await action_handler['func'](ctx, payload)
                    else:
                        logging.warning(f"No handler for action '{action_name}'")
            except ConnectionClosedError:
                logging.warning("Connection closed, attempting to reconnect...")
                break
            except Exception as e:
                logging.error(f"An error occurred: {e}")
                break

    async def _connect(self):
        while self._reconnect:
            try:
                async with connect(self.uri) as ws:
                    self._ws = ws
                    logging.info("WebSocket connection established.")

                    set_actions_message = {
                        "type": "registerActions",
                        "data": {
                            "actions": { action: handler['payload_description'] for action, handler in self._action_handlers.items() }
                        }
                    }
                    await self._ws.send(json.dumps(set_actions_message))

                    if 'on_ready' in self._event_handlers:
                        await self._event_handlers['on_ready']()
                    await self._handle_messages()
            except Exception as e:
                logging.error(f"An error occurred: {e}")
                logging.info(f"Reconnecting in {self._reconnect_interval} seconds...")
                await asyncio.sleep(self._reconnect_interval)

    def run(self):
        """
        Starts the connection to the smart building server.
        """
        asyncio.run(self._connect())
