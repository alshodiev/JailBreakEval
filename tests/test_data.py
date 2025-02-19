import json

file_path = "data/red_team_small.json"

with open(file_path, "r", encoding="utf-8") as f:
    data = json.load(f)  # Load full JSON array

# Print some sample data to verify
print(f"Loaded {len(data)} entries.")
print("First entry:", json.dumps(data[0], indent=4))