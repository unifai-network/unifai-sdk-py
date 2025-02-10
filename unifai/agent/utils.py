import datetime
import logging
import os
import yaml

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