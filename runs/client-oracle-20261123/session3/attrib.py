"""usage: attrib.py <id8> <rule-id|UNEXPLAINED|...> <stage> <text>  -> append/replace one fight attribution in session3/attribution.json"""
import json,sys
from pathlib import Path
p=Path(__file__).resolve().parent/'attribution.json';d=json.loads(p.read_text()) if p.exists() else {}
d[sys.argv[1]]=dict(rule=sys.argv[2],first_difference=sys.argv[3],note=sys.argv[4]);p.write_text(json.dumps(d,indent=1));print(len(d),'fights attributed')
