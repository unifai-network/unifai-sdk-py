import datetime
import logging
import os
import yaml
import uuid
import asyncio
from typing import Dict

logger = logging.getLogger(__name__)

def load_prompt(prompt_path):
    parts = prompt_path.split('.')
    prompt_name = parts[-1]
    file_name = '.'.join(parts[:-1])
    prompts = load_prompt_file(file_name)
    return prompts.get(prompt_name, '')

def load_prompt_file(file_name):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    prompts_path = os.path.join(script_dir, 'prompts', f'{file_name}.yaml')
    with open(prompts_path, 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)

def load_all_prompts():
    """
    Reads all YAML files in the prompts directory using the load_prompt_file function,
    loads their contents, and merges them into a single dictionary. The keys in the resulting
    dictionary are formatted as 'filename.key_in_file'.

    Returns:
        dict: A merged dictionary containing all key-value pairs from the YAML files.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    prompts_path = os.path.join(script_dir, 'prompts')
    all_prompts = {}
    
    for filename in os.listdir(prompts_path):
        if filename.endswith('.yaml'):
            file_name = os.path.splitext(filename)[0]
            try:
                data = load_prompt_file(os.path.join(prompts_path, file_name))
                if isinstance(data, dict):
                    for key, value in data.items():
                        all_prompts[f"{file_name}.{key}"] = value
                else:
                    logger.warning(f"{filename} does not contain a valid YAML dictionary.")
            except Exception as e:
                logger.error(f"Error reading {filename}: {e}")
    
    return all_prompts

def generate_uuid_from_id(id_str: str) -> uuid.UUID:
    """Generate a UUID from a string identifier."""
    return uuid.uuid5(uuid.NAMESPACE_DNS, str(id_str))

def get_collection_name(agent_id: str, client_id: str, chat_id: str) -> str:
    """Generate a unique collection name for memory storage"""
    collection_base = f"{agent_id}-{client_id}-{chat_id}"
    sanitized = ''.join(c if c.isalnum() else '-' for c in collection_base)
    if not sanitized[0].isalpha():
        sanitized = 'id-' + sanitized
    if len(sanitized) < 3:
        sanitized = sanitized + '-collection'
    elif len(sanitized) > 63:
        sanitized = sanitized[:60] + '-col'
    return sanitized

def sanitize_collection_name(id_str: str) -> str:
    sanitized = ''.join(c if c.isalnum() else '-' for c in id_str)
    if not sanitized[0].isalpha():
        sanitized = 'id-' + sanitized
    if len(sanitized) < 3:
        sanitized = sanitized + '-collection'
    elif len(sanitized) > 63:
        sanitized = sanitized[:60] + '-col'
    return sanitized

class ChannelLockManager:
    def __init__(self):
        self._channel_locks: Dict[str, asyncio.Lock] = {}
        
    def get_lock(self, client_id: str, chat_id: str) -> asyncio.Lock:
        """Get or create a lock for a specific channel"""
        channel_key = f"{client_id}:{chat_id}"
        if channel_key not in self._channel_locks:
            self._channel_locks[channel_key] = asyncio.Lock()
        return self._channel_locks[channel_key]