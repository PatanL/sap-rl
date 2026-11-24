from collections import Counter
import json
from case import details
def check():
 c=Counter();examples=[]
 for l in open('seats.jsonl'):
  row=json.loads(l)
  if row['split']!='dev':continue
  for k in row['states']:
   n=int(k)
   if n<2:continue
   try:d=details(row,n)
   except Exception:continue
   acts,logs=d['actions'],d['run']['log']
   for j in range(1,min(len(acts),len(logs)+1)):
    if 'freeze' not in acts[j]:continue
    actual=[i for i,flag in acts[j]['freeze']]
    offers=logs[j-1]['offers'];eng=[s[0] for s in offers];seen={s[0]:s for s in offers}
    c['all']+=1
    if set(eng)!=set(actual):
     c['not same IDs']+=1
     if len(examples)<5:examples.append([d['key'],j,eng,actual])
     continue
    opaque={s[0] for s in offers if s[2] in ('pet-none','None','food-none')}
    c['same IDs']+=1
    if actual==eng:c['same order']+=1
    if [i for i in actual if i not in opaque]==[i for i in eng if i not in opaque]:c['same resolved order']+=1
    if not opaque:
     c['none opaque']+=1;c['no opaque, same order']+=int(eng==actual)
 print(c,examples)
if __name__=='__main__':check()
