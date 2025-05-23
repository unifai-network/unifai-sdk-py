from __future__ import annotations
from datetime import datetime
import asyncio
import litellm
from litellm.exceptions import RateLimitError
import logging
import re
import traceback
import uuid
from typing import Dict, List

from .model import ModelManager
from .utils import (
    load_prompt,
    load_all_prompts,
    generate_uuid_from_id,
    get_collection_name,
    ChannelLockManager,
    sanitize_collection_name,
)
from ..common.const import BACKEND_WS_ENDPOINT
from ..memory import (
    ChromaConfig,
    ChromaMemoryManager,
    Memory,
    MemoryRole,
    MemoryType,
    StorageType,
    ToolInfo,
)
from ..reflector import FactReflector, GoalReflector
from ..tools import Tools
from ..client.base import BaseClient, MessageContext, Message

logger = logging.getLogger(__name__)


class Agent:
    def __init__(
        self,
        api_key: str,
        agent_id: str = "",
        clients: List[BaseClient] = [],
        chroma_config: ChromaConfig = ChromaConfig(
            storage_type=StorageType.PERSISTENT,
            persist_directory="./chroma_db",
        ),
        tool_call_concurrency: int = 10,
    ):
        """Initialize an Agent instance.

        Args:
            api_key (str): UnifAI agent API key
            agent_id (str, optional): Unique identifier for the agent. Different agent_ids will use
                separate memory collections in the database, allowing multiple agents to maintain
                independent conversation histories and memories using the same database.
            clients (List[BaseClient], optional): List of clients to handle different
                communication channels (e.g., Telegram)
            chroma_config (ChromaConfig, optional): Configuration for the Chroma database.
        """
        self.api_key = api_key
        self._agent_id = agent_id
        self._prompts = {}
        self._models = {
            "default": "anthropic/claude-3-7-sonnet-20250219",
        }
        self.set_ws_endpoint(BACKEND_WS_ENDPOINT)
        self.tools = Tools(api_key=self.api_key)
        self.model_manager = ModelManager()
        self._stop_event = asyncio.Event()
        self._tasks: List[asyncio.Task] = []
        self.model_timeout: float | None = 120

        self._channel_locks: Dict[str, asyncio.Lock] = {}

        self.fact_reflector = FactReflector(litellm.acompletion)
        self.goal_reflector = GoalReflector(litellm.acompletion)

        self.memory_config = chroma_config
        self.tool_call_concurrency = tool_call_concurrency

        self._clients: Dict[str, BaseClient] = {}
        if clients:
            for client in clients:
                self.add_client(client)

        self._channel_lock_manager = ChannelLockManager()

    def set_ws_endpoint(self, endpoint):
        self.ws_uri = f"{endpoint}?type=player&api-key={self.api_key}"

    def set_chat_completion_function(self, f):
        self.model_manager.set_chat_completion_function(f)

    def set_completion_cost_calculator(self, f):
        self.model_manager.set_completion_cost_calculator(f)

    def set_model_timeout(self, timeout: float | None):
        self.model_timeout = timeout

    def get_all_prompts(self):
        """
        Get all prompts used by combining default prompts with any custom prompts.

        Returns:
            dict: Combined dictionary of all prompts
        """
        prompts = load_all_prompts()
        prompts.update(self._prompts)
        return prompts

    def get_prompt(self, prompt_key):
        """
        Get a specific prompt by key, first checking custom prompts then default prompts.

        Args:
            prompt_key (str): Key of the desired prompt

        Returns:
            str: The prompt text
        """
        prompt = self._prompts.get(prompt_key)
        if not prompt:
            prompt = load_prompt(prompt_key)
        return prompt

    def set_prompt(self, prompt_key, prompt):
        """
        Set a custom prompt. Set to None or empty string to use default prompt.
        The prompt will be immediately effective even if agent is already running.

        Args:
            prompt_key (str): Key of the prompt
            prompt (str): The prompt text to set
        """
        self._prompts[prompt_key] = prompt

    def set_prompts(self, prompts):
        """
        Replace all custom prompts with a new dictionary.
        Missing keys from the provided dictionary will be loaded from default prompts.
        The prompts will be immediately effective even if agent is already running.

        Args:
            prompts (dict): Dictionary of prompts to set
        """
        self._prompts = prompts

    def update_prompts(self, prompts):
        """
        Update custom prompts by merging with provided prompts.
        The prompts will be immediately effective even if agent is already running.

        Args:
            prompts (dict): Dictionary of prompts to merge with existing prompts
        """
        self._prompts.update(prompts)

    def get_model(self, prompt_key):
        """
        Get the model for a specific prompt key, falling back to default model if not found.

        Args:
            prompt_key (str): Key to look up model for

        Returns:
            str: The model name to use
        """
        return self._models.get(prompt_key, self._models.get("default"))

    def set_model(self, prompt_key, model):
        """
        Set the model to use for a specific prompt key.
        Set the model of key 'default' to set the default model.
        The model will be immediately effective even if agent is already running.

        Args:
            prompt_key (str): Key to set model for
            model (str): Name of the model to use
        """
        self._models[prompt_key] = model

    def set_models(self, models):
        """
        Replace all custom models with a new dictionary.
        The model of key 'default' will be used as the default model.
        The models will be immediately effective even if agent is already running.

        Args:
            models (dict): Dictionary mapping prompt keys to model names
        """
        self._models = models

    def update_models(self, models):
        """
        Update custom models by merging with provided models.
        The model of key 'default' will be used as the default model.
        The models will be immediately effective even if agent is already running.

        Args:
            models (dict): Dictionary of models to merge with existing models
        """
        self._models.update(models)

    def run(self):
        """
        Synchronous entry point to run the agent.
        """
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self.start())
        except KeyboardInterrupt:
            logger.info("Agent interrupted by user. Shutting down...")
            loop.run_until_complete(self.stop())
        finally:
            pending = asyncio.all_tasks(loop)
            for task in pending:
                task.cancel()
            loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            loop.close()

    async def start(self):
        logger.info("Starting agent...")
        for client in self._clients.values():
            try:
                await client.start()
                self._tasks.append(
                    asyncio.create_task(self._handle_client_messages(client))
                )
                logger.info(f"Started client: {client.client_id}")
            except Exception as e:
                logger.error(f"Failed to start client {client.client_id}: {e}")

        await self._stop_event.wait()

        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)

        for client in self._clients.values():
            try:
                logger.info(f"Stopped client: {client.client_id}")
                await client.stop()
            except Exception as e:
                logger.error(f"Failed to stop client {client.client_id}: {e}")

        logger.info("Agent has been stopped.")

    async def stop(self):
        """Stop the agent"""
        logger.info("Stopping the agent...")
        self._stop_event.set()

    def get_memory_manager(self, user_id: str, chat_id: str) -> ChromaMemoryManager:
        chat_id = chat_id or user_id

        collection_base = f"{self._agent_id}-{user_id}-{chat_id}"
        collection_name = sanitize_collection_name(collection_base)

        config = ChromaConfig(
            storage_type=self.memory_config.storage_type,
            host=self.memory_config.host,
            port=self.memory_config.port,
            collection_name=collection_name,
        )
        return ChromaMemoryManager(config)

    def get_channel_lock(self, client_id: str, chat_id: str) -> asyncio.Lock:
        return self._channel_lock_manager.get_lock(client_id, chat_id)

    async def get_reply(
        self,
        client: BaseClient,
        ctx: MessageContext,
        history_count: int,
    ) -> tuple[list[Message], list[ToolInfo], list[Dict], tuple[int, int], float]:
        message = ctx.message
        user_id = ctx.user_id
        chat_id = ctx.chat_id
        memory_manager = self.get_memory_manager(user_id, chat_id)
        input_tokens = 0
        output_tokens = 0
        total_cost = 0

        response, cost = await self.model_manager.chat_completion(
            model=self.get_model("history"),
            messages=[
                {"role": "system", "content": self.get_prompt("agent.history")},
                {"role": "user", "content": message},
            ],
            timeout=self.model_timeout,
        )

        if response is not None:
            input_tokens += response.usage.prompt_tokens  # type: ignore
            output_tokens += response.usage.completion_tokens  # type: ignore
            total_cost += cost

            use_history = False
            try:
                logger.info(f"History response: {response.choices[0].message.content}")  # type: ignore
                history_score = int(response.choices[0].message.content)  # type: ignore
                use_history = history_score > 50
            except Exception as e:
                logger.error(f"Error determining whether to use history: {e}")
        else:
            logger.error("Failed to get history response, proceeding without history")
            use_history = False

        logger.info(f"Use history: {use_history}")

        recent_memories = []
        if use_history:
            try:
                recent_memories = await memory_manager.get_recent_memories(
                    count=history_count,
                )
            except Exception as e:
                logger.error(f"Error getting recent memories: {e}")
                use_history = False
                logger.info("Proceeding without history")

        relevant_memories = []
        try:
            relevant_memories = await memory_manager.get_memories(
                content=message,
                count=5,
                threshold=0.7,
                metadata={"type": {"$in": ["fact", "goal"]}},
            )
        except Exception as e:
            logger.error(f"Error getting relevant memories: {e}")
            logger.info("Proceeding without relevant memories")

        model = self.get_model("default") or ""
        anthropic_cache_control = model.lower().startswith("anthropic")

        system_prompt = self.get_prompt("agent.system").format(
            date=datetime.now().strftime("%Y-%m-%d"),
        )

        system_messages = [{"type": "text", "text": system_prompt}]

        if relevant_memories:
            facts = []
            goals = []
            for mem in relevant_memories:
                if mem.memory_type == MemoryType.FACT:
                    facts.extend(mem.content.get("claims", []))
                elif mem.memory_type == MemoryType.GOAL:
                    goals.extend(mem.content.get("goals", []))

            if facts:
                system_messages.append(
                    {
                        "type": "text",
                        "text": "Relevant facts:\n"
                        + "\n".join([f"- {fact}" for fact in facts]),
                    }
                )

            if goals:
                system_messages.append(
                    {
                        "type": "text",
                        "text": "Active goals:\n"
                        + "\n".join([f"- {goal}" for goal in goals]),
                    }
                )

        if anthropic_cache_control:
            system_messages[-1]["cache_control"] = {"type": "ephemeral"}
        messages: List = [{"role": "system", "content": system_messages}]

        if recent_memories:
            recent_interactions = sorted(
                recent_memories,
                key=lambda x: x.metadata.get("timestamp", ""),
                reverse=True,
            )[:history_count]

            recent_interactions = list(reversed(recent_interactions))

            for mem in recent_interactions:
                if mem.content.get("interaction", {}).get("messages"):
                    has_non_tool_message = False
                    for msg in mem.content["interaction"]["messages"]:
                        if msg.get("tool_calls"):
                            is_valid_tool_call = True
                            for tool_call in msg.get("tool_calls", []):
                                if not re.match(
                                    r"^[a-zA-Z0-9_-]{1,64}$",
                                    tool_call.get("function", {}).get("name", ""),
                                ):
                                    is_valid_tool_call = False
                            if not is_valid_tool_call:
                                continue
                        if msg.get("tool_call") != "tool":
                            has_non_tool_message = True
                            messages.append(msg)
                        elif has_non_tool_message:
                            messages.append(msg)

        messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": message,
                    }
                ],
            }
        )
        interaction = {"messages": [{"role": "user", "content": message}]}
        tool_infos_collection = []
        reply_messages = []

        sent_using_tools = False
        while True:
            if anthropic_cache_control:
                messages[-1]["content"][-1]["cache_control"] = {"type": "ephemeral"}
            response, cost = await self.model_manager.chat_completion(
                model=model,
                messages=messages,
                tools=await self.tools.get_tools(cache_control=anthropic_cache_control),
                parallel_tool_calls=True,
                extra_headers=(
                    {"anthropic-beta": "token-efficient-tools-2025-02-19"}
                    if anthropic_cache_control
                    else {}
                ),
                timeout=self.model_timeout,
            )
            if anthropic_cache_control:
                del messages[-1]["content"][-1]["cache_control"]

            if response is None:
                logger.error("Failed to get model response")
                break

            input_tokens += response.usage.prompt_tokens  # type: ignore
            output_tokens += response.usage.completion_tokens  # type: ignore
            total_cost += cost

            if not response.choices:  # type: ignore
                logger.error(f"Invalid response: {response}")
                break

            assistant_message = response.choices[0].message  # type: ignore

            if assistant_message.tool_calls:
                # Sanitize tool call function names to ensure they match the pattern
                # the tool call will still fail but at least the llm call will not fail so llm can correct itself
                for i, tool_call in enumerate(assistant_message.tool_calls):
                    if tool_call.function.name:
                        sanitized_name = re.sub(
                            r"[^a-zA-Z0-9_-]", "", tool_call.function.name
                        )
                        sanitized_name = sanitized_name[:64]
                        assistant_message.tool_calls[i].function.name = sanitized_name

            if assistant_message.content or assistant_message.tool_calls:
                messages.append(assistant_message)
                reply_messages.append(Message.model_validate(assistant_message.model_dump(mode="json")))
                interaction["messages"].append(
                    assistant_message.model_dump(mode="json")
                )

            if not assistant_message.tool_calls:
                break

            if ctx.progress_report and not sent_using_tools:
                ctx.cost = 0.0
                await client.send_message(
                    ctx,
                    [
                        Message(
                            role="assistant",
                            content="I'm working on it...",
                        )
                    ],
                )
                sent_using_tools = True

            for tool_call in assistant_message.tool_calls:
                tool_info = ToolInfo(
                    name=tool_call.function.name or "",
                    description=tool_call.function.arguments,
                )
                tool_infos_collection.append(tool_info)

            results = await self.tools.call_tools(assistant_message.tool_calls, concurrency=self.tool_call_concurrency)  # type: ignore

            if not results:
                break

            messages.extend(results)
            interaction["messages"].extend(results)
            for result in results:
                reply_messages.append(Message.model_validate(result))

            messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Above is the result",
                        }
                    ],
                }
            )

        return (
            reply_messages,
            tool_infos_collection,
            interaction["messages"],
            (input_tokens, output_tokens),
            total_cost,
        )

    async def process_message_with_memory(
        self,
        client: BaseClient,
        ctx: MessageContext,
        history_count: int = 1,
    ) -> tuple[List[Message], tuple[int, int]]:
        message = ctx.message
        user_id = ctx.user_id
        chat_id = ctx.chat_id
        memory_manager = self.get_memory_manager(user_id, chat_id)

        reply_messages, tool_infos, interaction_content, usage, cost = (
            await self.get_reply(client, ctx, history_count=history_count)
        )

        ctx.cost = cost
        await client.send_message(ctx, reply_messages)

        reply_text = reply_messages[-1].get("content", "") if reply_messages else ""

        user_uuid = generate_uuid_from_id(str(user_id))
        agent_uuid = generate_uuid_from_id(self._agent_id)

        base_metadata = {
            "chat_id": str(chat_id),
            "user_id": str(user_id),
            "timestamp": str(datetime.now().isoformat()),
            "has_tools": bool(tool_infos),
            "is_private": chat_id == user_id,
        }

        tasks = [
            self.fact_reflector.reflect(f"User: {message}\nAssistant: {reply_text}"),
            self.goal_reflector.reflect(f"User: {message}\nAssistant: {reply_text}"),
        ]

        fact_result, goal_result = await asyncio.gather(*tasks)

        memory_tasks = []

        if fact_result.success and fact_result.data and fact_result.data.get("claims"):
            metadata = base_metadata.copy()
            metadata.update(
                {"type": "fact", "claims_count": len(fact_result.data["claims"])}
            )
            if tool_infos:
                metadata["tool_names"] = ",".join(t.name for t in tool_infos)

            fact_memory = Memory(
                id=uuid.uuid4(),
                user_id=user_uuid,
                agent_id=agent_uuid,
                content={
                    "text": "Extracted facts from conversation",
                    "claims": fact_result.data["claims"],
                },
                memory_type=MemoryType.FACT,
                metadata=metadata,
                role=MemoryRole.SYSTEM,
                tools=tool_infos if tool_infos else [],
                unique=True,
            )
            memory_tasks.append(memory_manager.create_memory(fact_memory))

        if goal_result.success and goal_result.data and goal_result.data.get("goals"):
            metadata = base_metadata.copy()
            metadata.update(
                {"type": "goal", "goals_count": len(goal_result.data["goals"])}
            )
            if tool_infos:
                metadata["tool_names"] = ",".join(t.name for t in tool_infos)

            goal_memory = Memory(
                id=uuid.uuid4(),
                user_id=user_uuid,
                agent_id=agent_uuid,
                content={
                    "text": "Goals and progress tracking",
                    "goals": goal_result.data["goals"],
                },
                memory_type=MemoryType.GOAL,
                metadata=metadata,
                role=MemoryRole.SYSTEM,
                tools=tool_infos if tool_infos else [],
                unique=True,
            )
            memory_tasks.append(memory_manager.create_memory(goal_memory))

        metadata = base_metadata.copy()
        metadata.update({"type": "interaction", "message_length": len(message)})
        if tool_infos:
            metadata["tool_names"] = ",".join(t.name for t in tool_infos)
        interaction_memory = Memory(
            id=uuid.uuid4(),
            user_id=user_uuid,
            agent_id=agent_uuid,
            content={
                "text": f"User: {message}\nAssistant: {reply_text}",
                "interaction": {"messages": interaction_content},
            },
            memory_type=MemoryType.INTERACTION,
            metadata=metadata,
            role=MemoryRole.SYSTEM,
            tools=tool_infos if tool_infos else [],
            unique=False,
        )
        memory_tasks.append(memory_manager.create_memory(interaction_memory))

        if memory_tasks:
            await asyncio.gather(*memory_tasks)

        return reply_messages, usage

    def add_client(self, client: BaseClient) -> None:
        """Add a new client to the agent"""
        self._clients[client.client_id] = client

    def remove_client(self, client_id: str) -> None:
        """Remove a client from the agent"""
        if client_id in self._clients:
            del self._clients[client_id]

    async def _handle_client_messages(self, client: BaseClient) -> None:
        """Handle messages from a specific client with parallel processing between channels"""
        while not self._stop_event.is_set():
            try:
                ctx = await client.receive_message()
                if ctx:
                    self._tasks.append(
                        asyncio.create_task(self._process_channel_message(client, ctx))
                    )
            except Exception as e:
                logger.error(f"Error handling message from {client.client_id}: {e}")

    async def _process_channel_message(
        self, client: BaseClient, ctx: MessageContext
    ) -> None:
        """Process message within a channel"""
        channel_lock = self.get_channel_lock(client.client_id, ctx.chat_id)

        async with channel_lock:
            try:
                await self.process_message_with_memory(client, ctx)
            except Exception as e:
                error_traceback = traceback.format_exc()
                logger.error(
                    f"Error processing message in channel {ctx.chat_id}: {e}\n{error_traceback}"
                )
                error_message = "Sorry, something went wrong. Most likely the model is being rate limited due to high demand. Please try again later."
                if isinstance(e, RateLimitError):
                    error_message = "Sorry, I'm being rate limited due to high demand. Please try again later."
                ctx.cost = 0.0
                await client.send_message(
                    ctx,
                    [
                        Message(
                            role="assistant",
                            content=error_message,
                        )
                    ],
                )

    def get_collection_name(self, client_id: str, chat_id: str) -> str:
        """Generate a unique collection name for memory storage"""
        return get_collection_name(self._agent_id, client_id, chat_id)
