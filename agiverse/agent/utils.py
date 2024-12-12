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

def format_json(json_obj, indent=0):
    indent_str = '    ' * indent
    formatted_string = ""
    if isinstance(json_obj, dict):
        for key, value in json_obj.items():
            formatted_string += f"{indent_str}{key}:"
            if isinstance(value, (dict, list)):
                formatted_string += "\n" + format_json(value, indent + 1)
            else:
                formatted_string += f" {value}\n"
    elif isinstance(json_obj, list):
        for item in json_obj:
            formatted_string += indent_str + "- "
            if isinstance(item, (dict, list)):
                formatted_string += "\n" + format_json(item, indent + 1)
            else:
                formatted_string += f"{item}\n"
    else:
        formatted_string += f"{indent_str}{json_obj}\n"
    return formatted_string

def format_memory(memory):
    memory_str = ''
    for entry in memory:
        memory_str += f"Observation: {entry.get('observation', '')}\n"
        memory_str += f"Thought: {entry.get('thought', '')}\n"
        memory_str += f"Action: {format_json(entry.get('action', {}))}\n\n"
    return memory_str

def remove_additional_data(building):
    for key in ['boundary', 'rent']:
        building.pop(key, None)

def distance(x1, y1, x2, y2):
    return abs(x1 - x2) + abs(y1 - y2)

def distance_obj(obj1, obj2):
    if 'x' in obj1 and 'y' in obj1:
        x1, y1 = obj1['x'], obj1['y']
    elif 'locationX' in obj1 and 'locationY' in obj1:
        x1, y1 = obj1['locationX'], obj1['locationY']
    else:
        x1, y1 = 0, 0

    if 'x' in obj2 and 'y' in obj2:
        x2, y2 = obj2['x'], obj2['y']
    elif 'locationX' in obj2 and 'locationY' in obj2:
        x2, y2 = obj2['locationX'], obj2['locationY']
    else:
        x2, y2 = 0, 0

    return distance(x1, y1, x2, y2)

def is_valid_state_data(state_data):
    return (
        state_data 
        and isinstance(state_data, dict) 
        and 'playerID' in state_data
        and 'locationX' in state_data
        and 'locationY' in state_data
    )

def is_valid_player_data(players_data):
    return (
        players_data 
        and isinstance(players_data, list)
    )

def is_valid_map_data(map_data, key):
    return (
        map_data 
        and isinstance(map_data, dict) 
        and key in map_data 
        and isinstance(map_data[key], list)
    )

def seconds_since(time):
    return (datetime.datetime.now() - time).total_seconds()
