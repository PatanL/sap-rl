import sys,json,collections as C
from infer import *
from evidence import *
def main():
 out=[]
 for row,n,rec in jobs():
    p=prep(row,n);p['clauses']=stock_clauses(p);p['clues']=clues(row,n)
    bare,_=input_domains(p,False,False,False);cash,_=input_domains(p,False,True,False)
    receipts,_=input_domains(p,True,True,False);final,poss=input_domains(p)
    summoned=summon_certificate(p,poss)
    labels={i:next(iter(q)) for i,q in zip(p['b'],poss) if len(q)==1};labels.update(summoned)
    whole=(set(labels)==set(p['b']+p['e']) and not rec['skip'].startswith('unmapped') and final>0)
    source=None
    if whole:
      if summoned:source='Pill + documented summon/perk'
      elif cash==1 and bare!=1:source='sale proceeds'
      elif receipts==1 and cash!=1:source='named stock receipt + producer'
      elif final==1 and receipts!=1:source='recorded offer order + ledger'
      else:source='recorded UID / action constraints'
    out.append(dict(key=rec['key'],buy_ids=p['b'],endogenous_ids=p['e'],record_labels=labels,certified=whole,certificate=source,
        buy_count_pre_wallet=bare,buy_count_after_wallet=cash,buy_count_after_receipts=receipts,buy_count_after_order=final,
        receipt_clauses=p['clauses'],wallet=p['wallet'],offer_pairs=p['clues'][0],ordinary_uids=sorted(p['clues'][1]),
        buy_marginals=[sorted(q) for q in poss]))
 (P/'input-certificates.jsonl').write_text(''.join(json.dumps(x,separators=(',',':'))+'\n' for x in out))
 print(C.Counter(x['certificate'] for x in out))
if __name__=='__main__':main()
