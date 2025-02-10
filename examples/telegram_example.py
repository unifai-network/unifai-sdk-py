import logging
import os
import uuid
import unifai
import litellm
from telegram import Update, LinkPreviewOptions
from telegram.ext import filters, MessageHandler, ApplicationBuilder, ContextTypes
from dotenv import load_dotenv
from datetime import datetime
from uuid import UUID

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

class TelegramAgent(unifai.Agent):
    def __init__(self, api_key: str):
        super().__init__(api_key)
        self.fact_reflector = FactReflector(litellm)
        self.goal_reflector = GoalReflector(litellm)
        self._setup_memory_manager()
        self._setup_telegram()
        self._setup_chat_completion()
        self._agent_id = self.generate_agent_id()
        
    def _setup_memory_manager(self):
        self.memory_config = ChromaConfig(
            storage_type=StorageType.HTTP,
            host=os.getenv("CHROMA_HOST", "localhost"),
            port=int(os.getenv("CHROMA_PORT", "8000")),
        )
    
    def _setup_telegram(self):
        self.bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
        self.bot_name = os.getenv('TELEGRAM_BOT_NAME')
        self.channel_name = os.getenv('TELEGRAM_CHANNEL_NAME')
        if not self.bot_token:
            raise ValueError("TELEGRAM_BOT_TOKEN environment variable is required")
            
    def _setup_chat_completion(self):
        async def custom_chat_completion(messages, model='gpt-4o-mini', **kwargs):
            return await litellm.acompletion(
                model=model,
                messages=messages,
                **kwargs
            )
        self.set_chat_completion_function(custom_chat_completion)

    def get_memory_manager(self, user_id: str, chat_id: str = None) -> ChromaMemoryManager:
        is_private_chat = user_id == chat_id
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

        self._user_uuid = self.generate_uuid_from_id(user_id)
        self._chat_uuid = self.generate_uuid_from_id(chat_id)
        
        collection_name = sanitize_id(user_id if is_private_chat else chat_id)
        
        config = ChromaConfig(
            storage_type=self.memory_config.storage_type,
            host=self.memory_config.host,
            port=self.memory_config.port,
            collection_name=collection_name
        )
        return ChromaMemoryManager(config)

    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_message = update.message.text or update.message.caption
        if not user_message:
            return
        reply, launched = await self.process_message_with_memory(
            user_message,
            str(update.message.from_user.id),
            str(update.effective_chat.id),
        )

        await context.bot.send_message(
            chat_id=update.effective_chat.id,  
            text=reply, 
            link_preview_options=LinkPreviewOptions(is_disabled=True)
        )

    def generate_uuid_from_id(self, id_str: str) -> uuid.UUID:
        """Generate a UUID from a string identifier."""
        return uuid.uuid5(uuid.NAMESPACE_DNS, str(id_str))

    def generate_agent_id(self) -> uuid.UUID:
        """Generate a consistent agent ID."""
        return uuid.uuid5(uuid.NAMESPACE_DNS, f"telegram-bot-{self.bot_name}")

    async def get_reply(self, message: str, user_id: str, chat_id: str, history_count: int = 20) -> tuple[str, bool, list[ToolInfo]]:
        memory_manager = self.get_memory_manager(user_id, chat_id)
        tools = unifai.Tools(api_key=self.api_key)

        is_private_chat = user_id == chat_id
        recent_memories = []
        group_memories = []

        group_memories = await memory_manager.get_memories(
            content="",
            memory_type=MemoryType.INTERACTION,
            count=history_count,
            metadata={
                "user_id": str(user_id),
                "is_private": False
            }
        )

        private_memories = await memory_manager.get_memories(
            content="",
            memory_type=MemoryType.INTERACTION,
            count=history_count,
            metadata={
                "user_id": str(user_id),
                "is_private": True
            }
        )
        
        if is_private_chat:
            recent_memories = private_memories + group_memories
        else:
            recent_memories = group_memories
        
        relevant_memories = await memory_manager.get_memories(
            content=message,
            count=5, 
            threshold=0.7, 
            metadata={
                "user_id": str(user_id)
            }
        )

        recent_interactions = sorted(
            recent_memories,
            key=lambda x: x.metadata.get("timestamp", "")
        )[:history_count]
        
        messages = []
        system_prompt = self.get_prompt("agent.telegram") or """You are a helpful assistant with memory capabilities and access to various tools.
        You can recall relevant information from past conversations and use tools when needed.
        Always try to be helpful and informative while maintaining context from previous interactions."""
        
        messages.append({"content": system_prompt, "role": "system"})
        if relevant_memories:
            relevant_context = "Here are some relevant memories that might help:\n"
            for mem in relevant_memories:
                if mem.memory_type == MemoryType.FACT:
                    relevant_context += f"- Facts: {mem.content.get('claims', [])}\n"
                elif mem.memory_type == MemoryType.GOAL:
                    relevant_context += f"- Goals: {mem.content.get('goals', [])}\n"
                else:
                    relevant_context += f"- Previous interaction: {mem.content['text']}\n"
            messages.append({"content": relevant_context, "role": "system"})
        
        if recent_interactions:
            for mem in recent_interactions:
                content = mem.content['text']
                if content.startswith("User: "):
                    messages.append({"content": content[6:], "role": "user"})
                elif content.startswith("Assistant: "):
                    messages.append({"content": content[11:], "role": "assistant"})
        
        messages.append({"content": message, "role": "user"})
        interaction_content = []
        tool_infos_collection = []
        launched = False
        while True:
            model = self.get_model("agent.chat")
            def format_chat_messages(messages_list):
                formatted_messages = {
                    "messages": [
                        {
                            "role": msg["role"],
                            "content": msg["content"] if msg.get("content") else ""
                        }
                        for msg in messages_list
                    ],
                    "tools": tools.get_tools()
                }
                return formatted_messages

            formatted_chat = format_chat_messages(messages)
            response = await self.model_manager.chat_completion(
                model=model,
                **formatted_chat
            )
            
            assistant_message = response.choices[0].message
            
            if assistant_message.content:
                interaction_content.append(f"Assistant: {assistant_message.content}")
            
            message_to_append = {
                "role": "assistant",
                "content": assistant_message.content if assistant_message.content else ""
            }
            messages.append(message_to_append)
            
            if not assistant_message.tool_calls:
                break
                
            launched = True
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
                interaction_content.append(f"Tool result: {result['content']}")
                messages.append({
                    "role": "function",
                    "name": result.get("name", "unknown"),
                    "content": result.get("content", "")
                })
        
        reply = assistant_message.content if assistant_message.content else "I apologize, but I couldn't generate a proper response."
        
        return reply, launched, tool_infos_collection

    async def _start_telegram(self):
        while not self._stop_event.is_set():
            try:
                application = ApplicationBuilder().token(self.bot_token).build()
                
                application.add_handler(MessageHandler(
                    filters.CaptionRegex(f"@{self.bot_name}") & (~filters.COMMAND), 
                    self.handle_message
                ))
                application.add_handler(MessageHandler(
                    filters.Mention(self.bot_name) & (~filters.COMMAND), 
                    self.handle_message
                ))
                application.add_handler(MessageHandler(
                    filters.ChatType.PRIVATE & (~filters.COMMAND), 
                    self.handle_message
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
    ) -> tuple[str, bool]:
        is_private_chat = user_id == chat_id
        memory_manager = self.get_memory_manager(user_id, chat_id)

        reply, launched, tool_infos = await self.get_reply(
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
            "is_private": is_private_chat 
        }
        
        # Create memories with the updated metadata
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
        
        return reply, launched

def main():
    load_dotenv()
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    agent = TelegramAgent(api_key=os.getenv("AGENT_API_KEY"))
    
    try:
        agent.run()
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")

if __name__ == '__main__':
    main()