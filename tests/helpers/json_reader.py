import json
from pathlib import Path
import os

def get_json_from_file(folder_name, json_filename):
    with open(Path(os.path.abspath(folder_name), json_filename)) as f:
        return json.load(f)

def get_data_from_file(folder_name, json_filename):
    with open(Path(os.path.abspath(folder_name), json_filename)) as f:
        return f.read()

def write_data_to_file(folder_name, json_filename, data):
    with open(Path(os.path.abspath(folder_name), json_filename), 'w') as f:
        json.dump(f, data)