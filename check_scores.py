import json

with open('data/red_team_small.json', 'r') as f:
    data = json.load(f)
    
scores = [entry["min_harmlessness_score_transcript"] for entry in data]
print(scores)