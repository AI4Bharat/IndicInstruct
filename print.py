import os
import json

directory = "/sky-notebook/eval-results/"

for root, dirs, files in os.walk(directory):
    for file in files:
        if file.endswith('.json'):
            file_path = os.path.join(root, file)
            print(file_path)

            splits = file_path.replace(directory, "").split(['/'])
            task = splits[0]
            sub_task = splits[1]
            model = splits[2]
            shot = splits[3]
            file = splits[4]
            if file == "metrics.json":
                with open(file_path, 'r') as json_file:
                    try:
                        data = json.load(json_file)
                        print(f"Contents of {file}:")
                        print(json.dumps(data, indent=4))
                    except json.JSONDecodeError as e:
                        print(f"Error decoding JSON in {file}: {e}")