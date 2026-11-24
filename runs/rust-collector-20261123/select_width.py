import json
from pathlib import Path

def main():
    rates = {}
    for path in Path('results').glob('train-*/result.json'):
        record = json.loads(path.read_text())
        events = record['events']
        assert len(events)==24 and sum(x['decisions'] for x in events)==record['learned']
        first,last=events[3],events[-1]
        rates[record['games']]=sum(x['decisions'] for x in events[4:])/(last['seconds']-first['seconds'])
    if set(rates)!={128,192,256,384,512,1024}:raise ValueError('Incomplete sweep')
    best=max(rates,key=rates.get)
    Path('results/selection.json').write_text(json.dumps(dict(games=best,steady_learned_s=rates),indent=2))
    print(best)

if __name__=='__main__':main()
