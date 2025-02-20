from __future__ import annotations
from datetime import datetime
import asyncio
import json
import litellm
import logging
import os
import re
import uuid

from telegram import LinkPreviewOptions, Update
from telegram.ext import (
    ApplicationBuilder,
    ContextTypes,
    filters,
    MessageHandler,
)

from .model import ModelManager
from .utils import load_prompt, load_all_prompts
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
from ..tools.tools import FunctionName

logger = logging.getLogger(__name__)

class Agent:
    def __init__(self, api_key):
        self.api_key = api_key
        self._prompts = {}
        self._models = {
            'default': 'gpt-4o',
        }
        self.model_manager = ModelManager()
        self.set_ws_endpoint(BACKEND_WS_ENDPOINT)
        self._stop_event = asyncio.Event()
        self._tasks = []

        self.fact_reflector = FactReflector(litellm.acompletion)
        self.goal_reflector = GoalReflector(litellm.acompletion)
        self._setup_memory_manager()
        self._setup_telegram()
        self._agent_id = self.generate_agent_id()
        self._chat_locks = {}

    def set_ws_endpoint(self, endpoint):
        self.ws_uri = f"{endpoint}?type=player&api-key={self.api_key}"

    def set_chat_completion_function(self, f):
        self.model_manager.set_chat_completion_function(f)

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
        return self._models.get(prompt_key, self._models.get('default'))

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
        """
        Asynchronous entry point to run the agent.
        """
        self._tasks = [
            asyncio.create_task(self._start_telegram()),
        ]
        await self._stop_event.wait()
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        logger.info("Agent has been stopped.")

    async def stop(self):
        """
        Stop the agent.
        """
        logger.info("Stopping the agent...")
        self._stop_event.set()
        if hasattr(self, '_tasks'):
            await asyncio.gather(*self._tasks, return_exceptions=True)

    async def _start_telegram(self):
        while not self._stop_event.is_set():
            try:
                started = False
                application = ApplicationBuilder().token(self.bot_token).build()

                if not application.updater:
                    raise RuntimeError("Updater is not initialized")
                
                filter_conditions = (
                    (filters.CaptionRegex(f"@{self.bot_name}") & (~filters.COMMAND)) |
                    (filters.Mention(self.bot_name) & (~filters.COMMAND)) |
                    (filters.ChatType.PRIVATE & (~filters.COMMAND))
                )
                application.add_handler(MessageHandler(
                    filter_conditions,
                    self.handle_message,
                    block=False
                ))

                await application.initialize()
                await application.start()
                await application.updater.start_polling()
                started = True
                await self._stop_event.wait()
            except asyncio.CancelledError:
                logger.info("Telegram task cancelled.")
                break
            except Exception as e:
                logger.error(f"Error: {e}. Reconnecting in 5 seconds...")
                await asyncio.sleep(5)
            finally:
                if started and application.updater:
                    await application.updater.stop()
                    await application.stop()
                    await application.shutdown()

    def _setup_memory_manager(self):
        self.memory_config = ChromaConfig(
            storage_type=StorageType.HTTP,
            host=os.getenv("CHROMA_HOST", "localhost"),
            port=int(os.getenv("CHROMA_PORT", "8000")),
        )
    
    def _setup_telegram(self):
        self.bot_token = os.getenv('TELEGRAM_BOT_TOKEN', "")
        self.bot_name = os.getenv('TELEGRAM_BOT_NAME', "")
        if not self.bot_token:
            raise ValueError("TELEGRAM_BOT_TOKEN environment variable is required")

    def get_memory_manager(self, user_id: str, chat_id: str) -> ChromaMemoryManager:
        chat_id = chat_id or user_id
        
        def sanitize_id(id_str: str) -> str:
            sanitized = ''.join(c if c.isalnum() else '-' for c in id_str)
            if not sanitized[0].isalpha():
                sanitized = 'id-' + sanitized
            if len(sanitized) < 3:
                sanitized = sanitized + '-collection'
            elif len(sanitized) > 63:
                sanitized = sanitized[:60] + '-col'
            return sanitized

        collection_name = sanitize_id(chat_id)
        
        config = ChromaConfig(
            storage_type=self.memory_config.storage_type,
            host=self.memory_config.host,
            port=self.memory_config.port,
            collection_name=collection_name
        )
        return ChromaMemoryManager(config)

    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not update.message or not update.message.from_user or not update.effective_chat:
            return

        user_message = update.message.text or update.message.caption
        if not user_message:
            return
        reply = await self.process_message_with_memory(
            user_message,
            str(update.message.from_user.id),
            str(update.effective_chat.id),
        )

        MAX_MESSAGE_LENGTH = 4000
        messages = [reply[i:i + MAX_MESSAGE_LENGTH] for i in range(0, len(reply), MAX_MESSAGE_LENGTH)]
        
        for message in messages:
            await context.bot.send_message(
                chat_id=update.effective_chat.id,  
                text=message, 
                link_preview_options=LinkPreviewOptions(is_disabled=True)
            )

    def generate_uuid_from_id(self, id_str: str) -> uuid.UUID:
        """Generate a UUID from a string identifier."""
        return uuid.uuid5(uuid.NAMESPACE_DNS, str(id_str))

    def generate_agent_id(self) -> uuid.UUID:
        """Generate a consistent agent ID."""
        return uuid.uuid5(uuid.NAMESPACE_DNS, f"telegram-bot-{self.bot_name}")

    async def get_reply(self, message: str, user_id: str, chat_id: str, history_count: int = 10) -> tuple[str, list[ToolInfo], list[dict], list[dict]]:
        memory_manager = self.get_memory_manager(user_id, chat_id)
        tools = Tools(api_key=self.api_key)

        response = await self.model_manager.chat_completion(
            model=self.get_model("history"),
            messages=[
                {"role": "system", "content": self.get_prompt("agent.history")},
                {"role": "user", "content": message}
            ],
        )
        use_history = False
        try:
            logger.info(f"History response: {response.choices[0].message.content}")  # type: ignore
            history_score = int(response.choices[0].message.content)  # type: ignore
            use_history = history_score > 50
        except Exception as e:
            logger.error(f"Error determining whether to use history: {e}")

        logger.info(f"Use history: {use_history}")

        recent_memories = []
        if use_history:
            recent_memories = await memory_manager.get_recent_memories(
                count=history_count,
            )

        relevant_memories = await memory_manager.get_memories(
            content=message,
            count=5, 
            threshold=0.7,
            metadata={
                "type": {"$in": ["fact", "goal"]}
            }
        )

        messages = []
        system_prompt = self.get_prompt("agent.system") 
        messages.append({"content": system_prompt, "role": "system"})

        if relevant_memories:
            facts = []
            goals = []
            for mem in relevant_memories:
                if mem.memory_type == MemoryType.FACT:
                    facts.extend(mem.content.get('claims', []))
                elif mem.memory_type == MemoryType.GOAL:
                    goals.extend(mem.content.get('goals', []))
            
            if facts:
                messages.append({
                    "role": "system",
                    "content": "Relevant facts:\n" + "\n".join([f"- {fact}" for fact in facts])
                })
            
            if goals:
                messages.append({
                    "role": "system",
                    "content": "Active goals:\n" + "\n".join([f"- {goal}" for goal in goals])
                })

        if recent_memories:
            recent_interactions = sorted(
                recent_memories,
                key=lambda x: x.metadata.get("timestamp", ""),
                reverse=True
            )[:history_count]

            recent_interactions = list(reversed(recent_interactions))

            for mem in recent_interactions:
                if mem.content.get('interaction', {}).get('messages'):
                    has_non_tool_message = False
                    for msg in mem.content['interaction']['messages']:
                        if msg.get('tool_calls'):
                            is_valid_tool_call = True
                            for tool_call in msg.get('tool_calls', []):
                                if not re.match(r'^[a-zA-Z0-9_-]{1,64}$', tool_call.get('function', {}).get('name', '')):
                                    is_valid_tool_call = False
                            if not is_valid_tool_call:
                                continue
                        if msg.get('tool_call') != 'tool':
                            has_non_tool_message = True
                            messages.append(msg)
                        elif has_non_tool_message:
                            messages.append(msg)

        messages.append({"content": message, "role": "user"})
        interaction = {
            "messages": [
                {
                    "role": "user",
                    "content": message
                }
            ]
        }
        tool_infos_collection = []
        reply = ""
        
        while True:
            response = await self.model_manager.chat_completion(
                model=self.get_model("default"),
                messages=messages,
                tools=tools.get_tools(),
            )
            
            assistant_message = response.choices[0].message  # type: ignore

            if assistant_message.content or assistant_message.tool_calls:
                messages.append(assistant_message)
                interaction["messages"].append(assistant_message.model_dump(mode='json'))

            if assistant_message.content:
                reply += f'{assistant_message.content}\n'

            if not assistant_message.tool_calls:
                break
            
            for tool_call in assistant_message.tool_calls:
                tool_info = ToolInfo(
                    name=tool_call.function.name or "",
                    description=tool_call.function.arguments
                )
                tool_infos_collection.append(tool_info)
                
                args = json.loads(tool_call.function.arguments) if isinstance(tool_call.function.arguments, str) else tool_call.function.arguments
                if tool_call.function.name == FunctionName.SEARCH_TOOLS.value:
                    reply += f"Searching services: {args.get('query')}...\n"
                elif tool_call.function.name == FunctionName.CALL_TOOL.value:
                    reply += f"Invoking service: {args.get('action')}...\n"

            results = await tools.call_tools(assistant_message.tool_calls)  # type: ignore

            if not results:
                break

            messages.extend(results)
            interaction["messages"].extend(results)

        return (
            reply.rstrip('\n') if reply else "Sorry, something went wrong.",
            tool_infos_collection,
            messages,
            interaction["messages"]
        )

    async def process_message_with_memory(
        self,
        message: str,
        user_id: str,
        chat_id: str,
        history_count: int = 10,
    ) -> str:
        if chat_id not in self._chat_locks:
            self._chat_locks[chat_id] = asyncio.Lock()
        
        async with self._chat_locks[chat_id]:
            memory_manager = self.get_memory_manager(user_id, chat_id)
            
            reply, tool_infos, messages, interaction_content = await self.get_reply(
                message, 
                user_id, 
                chat_id, 
                history_count=history_count
            )
            
            user_uuid = self.generate_uuid_from_id(str(user_id))
            
            fact_result = await self.fact_reflector.reflect(f"User: {message}\nAssistant: {reply}")
            goal_result = await self.goal_reflector.reflect(f"User: {message}\nAssistant: {reply}")
            
            base_metadata = {
                "chat_id": str(chat_id),
                "user_id": str(user_id), 
                "timestamp": str(datetime.now().isoformat()),
                "has_tools": bool(tool_infos),
                "is_private": chat_id == user_id,
            }
            
            if fact_result.success and fact_result.data and fact_result.data.get('claims'):
                metadata = base_metadata.copy()
                metadata.update({
                    "type": "fact",
                    "claims_count": len(fact_result.data['claims'])
                })
                if tool_infos:
                    metadata["tool_names"] = ",".join(t.name for t in tool_infos)

                fact_memory = Memory(
                    id=uuid.uuid4(),
                    user_id=user_uuid,
                    agent_id=self._agent_id,
                    content={
                        "text": "Extracted facts from conversation",
                        "claims": fact_result.data['claims']
                    },
                    memory_type=MemoryType.FACT,
                    metadata=metadata,
                    role=MemoryRole.SYSTEM,
                    tools=tool_infos if tool_infos else [],
                    unique=True
                )
                await memory_manager.create_memory(fact_memory)
                
            if goal_result.success and goal_result.data and goal_result.data.get('goals'):
                metadata = base_metadata.copy()
                metadata.update({
                    "type": "goal",
                    "goals_count": len(goal_result.data['goals'])
                })
                if tool_infos:
                    metadata["tool_names"] = ",".join(t.name for t in tool_infos)

                goal_memory = Memory(
                    id=uuid.uuid4(),
                    user_id=user_uuid,
                    agent_id=self._agent_id,
                    content={
                        "text": "Goals and progress tracking",
                        "goals": goal_result.data['goals']
                    },
                    memory_type=MemoryType.GOAL,
                    metadata=metadata,
                    role=MemoryRole.SYSTEM,
                    tools=tool_infos if tool_infos else [],
                    unique=True
                )
                await memory_manager.create_memory(goal_memory)
                
            metadata = base_metadata.copy()
            metadata.update({
                "type": "interaction",
                "message_length": len(message)
            })
            if tool_infos:
                metadata["tool_names"] = ",".join(t.name for t in tool_infos)
            interaction_memory = Memory(
                id=uuid.uuid4(),
                user_id=user_uuid,
                agent_id=self._agent_id,
                content={
                    "text": f"User: {message}\nAssistant: {reply}",
                    "interaction": {
                        "messages": interaction_content
                    }
                },
                memory_type=MemoryType.INTERACTION,
                metadata=metadata,
                role=MemoryRole.SYSTEM,
                tools=tool_infos if tool_infos else [],
                unique=False
            )
            await memory_manager.create_memory(interaction_memory)
            
            return reply
