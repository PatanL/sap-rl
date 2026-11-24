import json,collections as C
from rendered_boards import *
def main():
    truth=json.load(open(D/'validation-results.json'));localmap={};g=[]
    for j in truth:
        dest=D/'captures'/j['fight'];a,_=anchors(dest);localmap[j['fight']]=a;g.extend(a)
    gn,gv,_,_=bank_arrays(g);counts=C.Counter();out=[]
    for j in truth:
        dest=D/'captures'/j['fight'];ln,lv,_,_=bank_arrays(localmap[j['fight']]);r=json.load(open(dest/'run.json'));b=json.load(open(D/'fixtures'/r['fixture']))
        im=cv2.imread(str(dest/r['terminal_frame']));slots=badges(im);actual=[]
        reach={p[0] for side in (0,1) for p in ordered_input(b,side)}|{'Bus','Bee','Ram','Chick','Zombie Cricket','Dirty Rat','Zombie Fly'}
        if 'Spider' in reach:reach.update(['Badger','Camel','Dodo','Dog','Dolphin','Elephant','Giraffe','Ox','Rabbit','Sheep'])
        for z in slots:
            f=compact(feature(im,z));n=predict(f,ln,lv,.095,.020) or predict(f,gn,gv,.069,.024)
            actual.append([n if n in reach else None,z['attack'],z['health']])
        real=j['read'];counts['manual_survivors']+=len(real);counts['slot_count_match']+=len(slots)==len(real)
        if len(real)==len(slots):
            for a,e in zip(actual,real):
                counts['aligned']+=1
                counts['species_named']+=a[0]!=None;counts['species_correct']+=a[0]==e[0]
                counts['stats_read']+=None not in a[1:];counts['stats_correct']+=a[1:]==e[1:]
        out.append(dict(fight=j['fight'],predicted=actual,manual=real))
    (D/'automation-validation.json').write_text(json.dumps(dict(counts=counts,rows=out),indent=2));print(counts)
if __name__=='__main__':main()
