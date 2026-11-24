from pathlib import Path
import sys,json
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import extract
out=Path(__file__).resolve().parent/'fresh'
extract.raw=extract.P.parent/'replay-oracle-20261122/raw-fresh'
extract.archive=out/'no-archive'
extract.P=out
extract.extract()
