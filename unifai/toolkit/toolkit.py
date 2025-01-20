import asyncio
import inspect
import json
import logging
from enum import Enum
from typing import Callable, Dict, Any, Optional
from pydantic import ValidationError, BaseModel
from websockets import connect, ConnectionClosedError
from websockets.asyncio.client import ClientConnection
from ..common.const import FRONTEND_API_ENDPOINT, BACKEND_WS_ENDPOINT
from .api import ToolkitAPI
from .context import ActionContext, ActionResult
from .messages import ActionDescription, ServerToToolkitMessage, ServerToToolkitMessageType, ActionMessageData, ToolkitToServerMessage, ToolkitToServerMessageType, RegisterActionsMessageData

logger = logging.getLogger(__name__)

EventHandler = Callable[..., Any]
ActionHandlerFunc = Callable[..., Optional[ActionResult]]

class EventType(Enum):
    ON_READY = "on_ready"

class ActionHandler(BaseModel):
    func: ActionHandlerFunc
    action_description: str | dict
    payload_description: str | dict 
    payment_description: str | dict

    class Config:
        arbitrary_types_allowed = True

class Toolkit:
    _api_key: str
    _event_handlers: Dict[EventType, EventHandler] = {}
    _action_handlers: Dict[str, ActionHandler] = {}
    _ws_uri: str
    _ws: Optional[ClientConnection] = None
    _reconnect_interval: float = 5
    _reconnect: bool = True
    _api: ToolkitAPI

    def __init__(self, api_key: str, reconnect_interval: float = 5):
        """
        A collection of tools (actions) that can be searched and called by agents.

        :param api_key: the API key of your toolkit.
        :param reconnect_interval: Time in seconds between reconnection attempts.
        """
        self._api_key = api_key
        self._reconnect_interval = reconnect_interval
        self._api = ToolkitAPI(api_key)
        self.set_api_endpoint(FRONTEND_API_ENDPOINT)
        self.set_ws_endpoint(BACKEND_WS_ENDPOINT)

    def set_ws_endpoint(self, endpoint: str):
        self._ws_uri = f"{endpoint}?type=toolkit&api-key={self._api_key}"

    def set_api_endpoint(self, endpoint: str):
        self._api.set_endpoint(endpoint)

    def event(self, func: EventHandler) -> EventHandler:
        """
        Decorator to register an event handler.
        """
        try:
            self._event_handlers[EventType(func.__name__)] = func
        except Exception as e:
            logger.error(f"{func.__name__} is not a valid event type and will be ignored")
        return func

    def action(self, action: str, action_description: str | dict = '', payload_description: str | dict = '', payment_description: str | dict = '') -> Callable[[ActionHandlerFunc], ActionHandlerFunc]:
        """
        Decorator to register an action handler.
        """
        def decorator(func: ActionHandlerFunc) -> ActionHandlerFunc:
            self._action_handlers[action] = ActionHandler(
                func=func,
                action_description=action_description,
                payload_description=payload_description,
                payment_description=payment_description
            )
            return func
        return decorator

    async def update_toolkit(self, name: Optional[str] = None, description: Optional[str] = None):
        """
        Update the toolkit name and/or description.
        """
        await self._api.update_toolkit(name=name, description=description)

    async def _handle_action(self, action_data: ActionMessageData):
        action_name = action_data.action
        action_handler = self._action_handlers.get(action_name)
        if action_handler:
            ctx = ActionContext(
                toolkit=self,
                agent_id=action_data.agentID,
                action_id=action_data.actionID,
                action_name=action_name
            )

            payload = action_data.payload or {}
            payment = action_data.payment

            if isinstance(payload, str):
                try:
                    payload = json.loads(payload)
                except Exception as e:
                    pass

            num_params = len(inspect.signature(action_handler.func).parameters)
            try:
                args = [ctx, payload, payment][:num_params]
                if asyncio.iscoroutinefunction(action_handler.func):
                    result = await action_handler.func(*args)
                else:
                    result = action_handler.func(*args)
                if not result:
                    result = ctx.Result(None)
                    logger.warning(f"Action handler '{action_name}' returned None, sending empty result. It is recommended to return a result from the action handler.")
                await ctx.send_result(result)
            except Exception as e:
                logger.error(f"An error occurred while handling action '{action_name}', please consider adding error handling and notify the caller: {e}")
                await ctx.send_result(ctx.Result({"error": "An unexpected error occurred, please report to the smart tool developer"}))
        else:
            logger.warning(f"No handler for action '{action_name}'")

    async def _handle_messages(self):
        while True:
            try:
                message = await self._ws.recv()

                logger.debug(f"Received raw message: {message}")

                try:
                    msg = ServerToToolkitMessage.model_validate_json(message)
                except json.JSONDecodeError:
                    logger.warning("Received a non-JSON message.")
                    continue
                except ValidationError as e:
                    logger.warning(f"Message validation error: {e}")
                    continue

                if msg.type == ServerToToolkitMessageType.ACTION:
                    try:
                        action_data = ActionMessageData(**msg.data)
                    except ValidationError as e:
                        logger.warning(f"Action message validation error: {e}")
                        continue
                    await self._handle_action(action_data)
            except ConnectionClosedError:
                logger.warning("Connection closed, attempting to reconnect...")
                break
            except Exception as e:
                logger.warning(f"An error occurred: {e}")
                break

    async def _connect(self):
        while self._reconnect:
            try:
                async with connect(self._ws_uri) as ws:
                    self._ws = ws

                    logger.info("WebSocket connection established.")

                    actions_data = {
                        action: ActionDescription(
                            description=handler.action_description,
                            payload=handler.payload_description,
                            payment=handler.payment_description
                        )
                        for action, handler in self._action_handlers.items()
                    }

                    set_actions_message = ToolkitToServerMessage(
                        type=ToolkitToServerMessageType.REGISTER_ACTIONS,
                        data=RegisterActionsMessageData(actions=actions_data).model_dump(),
                    )

                    await self._ws.send(set_actions_message.model_dump_json())

                    if EventType.ON_READY in self._event_handlers:
                        await self._event_handlers[EventType.ON_READY]()

                    await self._handle_messages()
            except Exception as e:
                logger.error(f"An error occurred: {e}")
                logger.info(f"Reconnecting in {self._reconnect_interval} seconds...")
                await asyncio.sleep(self._reconnect_interval)

    async def run(self):
        """
        Starts serving the toolkit.
        """
        await self._connect()
