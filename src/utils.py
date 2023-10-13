import yaml
import json
import re


def load_dataset(json_path):
    return [json.loads(line) for line in open(json_path, 'r')]


def load_config(path):
    """
    Load config from a YAML file
    :param path:
    :return: config dict
    """
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    return config


def save_dataset(self, path):
    with open(path, 'w') as f:
        json.dump(self.dataset, f)


def clean_instruction(instruction):
    # Remove everything before "Prompt#:"
    instruction = instruction.split("Prompt#:")[-1]

    # Remove specific prefixes
    instruction = re.sub(r'^(New Prompt:\s?)', '', instruction)

    # Replace certain phrases referring to the AI
    replacements = {
        r'As an AI (assistant|language model), I': 'I',
        r'As an AI (assistant|language model), you': 'You',
        r'As an AI (assistant|language model), what': 'What'
    }

    for pattern, replacement in replacements.items():
        instruction = re.sub(pattern, replacement, instruction)

    # Strip any leading or trailing whitespaces
    return instruction.strip()
