import os
import json
import requests
from datetime import datetime, timedelta

class DataTypes:
    MODEL_RESPONSE = 'model_response'
    SERVER_MESSAGE = 'server_message'
    SYSTEM_MESSAGE = 'system_message'

def load_file(filename):
    with open(filename, 'r') as f:
        return json.load(f)

def save_file(content, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w') as f:
        f.write(content)

def save_image(url, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    response = requests.get(url)
    response.raise_for_status()
    with open(filename, 'wb') as f:
        f.write(response.content)

def save_data(data_dir, name, data_type, data, memory_manager=None):
    os.makedirs(data_dir, exist_ok=True)
    with open(f'{data_dir}/{name}.jsonl', 'a') as f:
        json.dump({
            'type': data_type,
            'data': data,
            'timestamp': datetime.now().isoformat()
        }, f)
        f.write('\n')
    
async def save_to_memory(data, memory_manager):
    if not memory_manager:
        return
    
    msg_type = data.get('type')
    if msg_type == 'actionResult':
        action_data = data.get('data', {})
        action_result = f"{action_data.get('payload', {})}"
        action_id = action_data.get('actionID')
        
        all_memories = await memory_manager.memory_stream.get_all_memories()
        for memory in all_memories:
            metadata = memory.metadata
            if not memory.metadata:
                memory.metadata = {}
                metadata = memory.metadata
            if memory.type == 'actionResult' and metadata.get('action_id') == action_id:
                try:  
                    content_json = json.loads(metadata.get('original_content').rstrip('.'))
                    content_json['actionResult'] = action_result
                    metadata['original_content']= json.dumps(content_json)
                    await memory_manager.memory_stream.update_memory(memory)
                except json.JSONDecodeError:
                    print(f"Error: Could not parse original_content as JSON: {memory.original_content}")
                break
    elif msg_type == 'chat':
        chat_data = data.get('data', {})
        content = f"{chat_data.get('message')}"
        metadata = {
            'sender_id': str(chat_data.get('senderPlayerID')),
        }
        await memory_manager.add_memory(
            content=content,
            memory_type='chat',
            metadata=metadata,
            associated_agents=[chat_data.get('senderPlayerID')]
        )
    elif msg_type == 'system':
        system_data = data.get('data', {})
        content = f"{system_data.get('message')}"
        channel_id = system_data.get('channelID')
        metadata = {
            'channel_id': channel_id,
        }
        await memory_manager.add_memory(
            content=content,
            memory_type='system',
            metadata=metadata,
            associated_agents=[]
        )

def load_data(data_dir, name, data_type=None, since=None, max_responses=None):
    with open(f'{data_dir}/{name}.jsonl', 'r') as f:
        data = [json.loads(line) for line in f if line.strip()]
    
    if data_type is not None:
        data = [d for d in data if d['type'] == data_type]
    
    if since is not None:
        if isinstance(since, (int, float)):
            since_dt = datetime.now() - timedelta(hours=since)
        elif isinstance(since, str):
            since_dt = datetime.fromisoformat(since)
        else:
            since_dt = since
        data = [d for d in data if datetime.fromisoformat(d['timestamp']) >= since_dt]
    
    if max_responses is not None:
        data = data[-max_responses:]

    return data
