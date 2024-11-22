import datetime
import os
import yaml

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
                    print(f"Warning: {filename} does not contain a valid YAML dictionary.")
            except Exception as e:
                print(f"Error reading {filename}: {e}")
    
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

def is_nearby_building(building, state_data, distance_threshold):
    entrance = building.get('entrance', {})
    if not entrance:
        return False
    dx = abs(entrance.get('x', 0) - state_data.get('locationX', 0))
    dy = abs(entrance.get('y', 0) - state_data.get('locationY', 0))
    return dx + dy <= distance_threshold

def nearest_house(buildings, state_data):
    nearest = None
    min_distance = float('inf')
    for building in buildings:
        if building.get('type') == 'house':
            dx = abs(building['entrance']['x'] - state_data['locationX'])
            dy = abs(building['entrance']['y'] - state_data['locationY'])
            distance = dx + dy
            if distance < min_distance:
                min_distance = distance
                nearest = building
    return nearest

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

def is_nearby_player(player, state_data, distance_threshold):
    if player.get('playerID') == state_data.get('playerID'):
        return False
    dx = abs(player.get('locationX', 0) - state_data.get('locationX', 0))
    dy = abs(player.get('locationY', 0) - state_data.get('locationY', 0))
    return dx <= distance_threshold and dy <= distance_threshold

def seconds_since(time):
    return (datetime.datetime.now() - time).total_seconds()
