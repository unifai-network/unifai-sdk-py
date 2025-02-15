import sys
sys.path.append('.')

import asyncio
import logging
import os
import uuid
import unifai
import litellm
from telegram import Update, LinkPreviewOptions
from telegram.ext import filters, MessageHandler, ApplicationBuilder, ContextTypes
from dotenv import load_dotenv
from datetime import datetime
from unifai.memory import (
    Memory, 
    ChromaConfig,
    ChromaMemoryManager,
    StorageType,
    MemoryRole,
    MemoryType,
    ToolInfo
)
from unifai.reflector import FactReflector, GoalReflector
import json

load_dotenv()
    
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
    
class TelegramAgent(unifai.Agent):
    def __init__(self, api_key: str):
        super().__init__(api_key)
        self.fact_reflector = FactReflector(litellm.acompletion)
        self.goal_reflector = GoalReflector(litellm.acompletion)
        self._setup_memory_manager()
        self._setup_telegram()
        self._setup_chat_completion()
        self._agent_id = self.generate_agent_id()
        self._chat_locks = {}
        
    def _setup_memory_manager(self):
        self.memory_config = ChromaConfig(
            storage_type=StorageType.HTTP,
            host=os.getenv("CHROMA_HOST", "localhost"),
            port=int(os.getenv("CHROMA_PORT", "8000")),
        )
    
    def _setup_telegram(self):
        self.bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
        self.bot_name = os.getenv('TELEGRAM_BOT_NAME')
        if not self.bot_token:
            raise ValueError("TELEGRAM_BOT_TOKEN environment variable is required")
            
    def _setup_chat_completion(self):
        async def custom_chat_completion(messages, model, tools=None, **kwargs):
            return await litellm.acompletion(
                model=model,
                messages=messages,
                tools=tools,
                **kwargs
            )
        self.set_chat_completion_function(custom_chat_completion)

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
        if not update.message:
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

    async def get_reply(self, message: str, user_id: str, chat_id: str, history_count: int = 20) -> tuple[str, list[ToolInfo]]:
        memory_manager = self.get_memory_manager(user_id, chat_id)
        tools = unifai.Tools(api_key=self.api_key)

        recent_memories = await memory_manager.get_recent_memories(
            count=history_count,
            metadata={
                "chat_id": str(chat_id),
            }
        )

        relevant_memories = await memory_manager.get_memories(
            content=message,
            count=5, 
            threshold=0.7,
            metadata={
                "chat_id": str(chat_id),
                "type": {"$in": ["fact", "goal"]}
            }
        )

        messages = []
        system_prompt = self.get_prompt("agent.telegram") or "..."
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
                    "content": f"Related facts:\n{json.dumps(facts, indent=2)}"
                })
            if goals:
                messages.append({
                    "role": "system", 
                    "content": f"Active goals:\n{json.dumps(goals, indent=2)}"
                })

        if recent_memories:
            recent_interactions = sorted(
                recent_memories,
                key=lambda x: x.metadata.get("timestamp", "")
            )[:history_count]
            
            for mem in recent_interactions:
                if mem.content.get('messages'):
                    messages.extend(mem.content['messages'])

        messages.append({"content": message, "role": "user"})
        
        interaction_content = []
        tool_infos_collection = []
        reply = ""

        while True:
            response = await self.model_manager.chat_completion(
                model='gpt-4o',
                messages=messages,
                tools=tools.get_tools(),
            )
            
            assistant_message = response.choices[0].message
            
            if assistant_message.content:
                reply += f'{assistant_message.content}\n'
                interaction_content.append({
                    "role": "assistant",
                    "content": assistant_message.content
                })

            messages.append(assistant_message)
            
            if not assistant_message.tool_calls:
                break
                
            tool_infos = [
                ToolInfo(
                    name=tool_call.function.name,
                    description=tool_call.function.arguments
                )
                for tool_call in assistant_message.tool_calls
            ]
            tool_infos_collection.extend(tool_infos)
            
            results = await tools.call_tools(assistant_message.tool_calls)
            
            if not results:
                break
            
            for result in results:
                interaction_content.append({
                    "role": "tool",
                    "tool_call_id": result["tool_call_id"],
                    "content": result["content"]
                })
                messages.append(result)
        memory = Memory(
            id=uuid.uuid4(),
            user_id=self.generate_uuid_from_id(str(user_id)),
            agent_id=self._agent_id,
            content={
                "text": f"User: {message}\nAssistant: {reply}",
                "messages": messages,
                "context": {
                    "user_message": message,
                    "messages": interaction_content
                }
            },
            memory_type=MemoryType.INTERACTION,
            metadata={
                "chat_id": str(chat_id),
                "timestamp": str(datetime.now().isoformat()),
                "has_tools": bool(tool_infos_collection)
            },
            role=MemoryRole.SYSTEM,
            tools=tool_infos_collection if tool_infos_collection else [],
            unique=False
        )
        await memory_manager.create_memory(memory)
        
        return reply.rstrip('\n') if reply else "Sorry, something went wrong.", tool_infos_collection

    async def _start_telegram(self):
        while not self._stop_event.is_set():
            try:
                application = ApplicationBuilder().token(self.bot_token).build()
                
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

                started = False
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
                if started:
                    await application.updater.stop()
                    await application.stop()
                    await application.shutdown()

    async def process_message_with_memory(
        self,
        message: str,
        user_id: str,
        chat_id: str,
        history_count: int = 20,
    ) -> str:
        # Get or create a lock for this chat_id
        if chat_id not in self._chat_locks:
            self._chat_locks[chat_id] = asyncio.Lock()
        
        # Acquire the lock for this chat_id
        async with self._chat_locks[chat_id]:
            memory_manager = self.get_memory_manager(user_id, chat_id)
            
            reply, tool_infos = await self.get_reply(
                message, 
                user_id, 
                chat_id, 
                history_count=history_count
            )
            
            interaction_content = f"User: {message}\nAssistant: {reply}"
            user_uuid = self.generate_uuid_from_id(str(user_id))
            
            fact_result = await self.fact_reflector.reflect(interaction_content)
            goal_result = await self.goal_reflector.reflect(interaction_content)
            
            base_metadata = {
                "chat_id": str(chat_id),
                "user_id": str(user_id), 
                "timestamp": str(datetime.now().isoformat()),
                "has_tools": bool(tool_infos),
                "is_private": chat_id == user_id,
            }
            
            if fact_result.success and fact_result.data.get('claims'):
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
                
            if goal_result.success and goal_result.data.get('goals'):
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
                    "text": interaction_content
                },
                memory_type=MemoryType.INTERACTION,
                metadata=metadata,
                role=MemoryRole.SYSTEM,
                tools=tool_infos if tool_infos else [],
                unique=False
            )
            await memory_manager.create_memory(interaction_memory)
            
            return reply

def main():
    agent = TelegramAgent(api_key=os.getenv("AGENT_API_KEY"))
    
    try:
        agent.run()
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")

if __name__ == '__main__':
    main()