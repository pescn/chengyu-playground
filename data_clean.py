import json

with open("idiom.json") as f:
    data = json.load(f)

chengyus = [chengyu["word"] for chengyu in data]

with open("chengyu.json", "w") as f:
    json.dump(chengyus, f, ensure_ascii=False)