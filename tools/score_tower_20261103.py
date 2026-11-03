import csv, itertools, numpy as np
from jenga_vision import ends
shots={}
for s in ['captures/autoscan_20261103_114143','captures/autoscan_20261103_115306']:
    for r in csv.DictReader(open(s+'/ends.csv')): shots[r['key']]=float(r['gap'])
P='빼안빼'; M='안빼안'
truth_l={2:P,4:M,6:M,8:P,10:M,12:P,14:M}
B=('left','middle','right')
truth={(l,b):truth_l[l][i]=='빼' for l,t in truth_l.items() if l in truth_l for i,b in enumerate(B)}
LAY=[4,6,8,10,12,14]
print('base rate',sum(truth[(l,b)] for l in LAY for b in B),'of',len(LAY)*3)
def rel(method,layers):
    return [(sc,l,b) for l in layers for b,sc in ends.block_scores(shots,l,2,method).items()]
def raw(method,layers):
    out=[]
    for l in layers:
        g=ends.end_gaps(shots,l,2); 
        for b in B:
            v=[]
            if method in('near','mean') and b in g['near']: v.append(g['near'][b])
            if method in('far','mean') and b in g['far']: v.append(g['far'][b])
            if v: out.append((float(np.mean(v)),l,b))
    return out
def play(order,rule,tr):
    seq=[];done=set();i=0;q=list(order);forced=None
    while True:
        if forced: l,b=forced;forced=None
        else:
            q=[x for x in q if (x[1],x[2]) not in done]
            if not q: return seq,True
            _,l,b=q[0]
        if (l,b) in done: continue
        done.add((l,b)); seq.append((l,b,tr[(l,b)]))
        if not tr[(l,b)]: return seq,False
        if rule:
            if b=='middle': done|={(l,'left'),(l,'right')}
            else:
                o='right' if b=='left' else 'left'; forced=(l,o); done.add((l,'middle'))
def run(items,rule,tr):
    # ties: best / worst / default(upper layer first, as the code does)
    d=sorted(items,key=lambda x:(-x[0],-x[1]))
    s,_=play(d,rule,tr); sd=sum(x[2] for x in s)
    # groups by tie
    vals=sorted({x[0] for x in items},reverse=True)
    best=lambda o:sorted(o,key=lambda x:(-x[0],-tr[(x[1],x[2])],-x[1])); worst=lambda o:sorted(o,key=lambda x:(-x[0],tr[(x[1],x[2])],-x[1]))
    sb,_=play(best(items),rule,tr); sw,_=play(worst(items),rule,tr)
    return sd,sum(x[2] for x in sw),sum(x[2] for x in sb),s
for tname,tr,lay in [('L16 제외',truth,LAY),('L16 전부 성공 취급',{**truth,**{(16,b):True for b in B}},LAY+[16])]:
    print('=====',tname)
    for fn,name in [(rel,'relative'),(raw,'raw px')]:
        print('--',name)
        for m,lab in [('near','front'),('far','back'),('mean','both')]:
            it=fn(m,lay)
            for rule in (False,True):
                d,w,b,s=run(it,rule,tr)
                print(f'{lab:5} rule={rule!s:5} streak {d} (worst {w} best {b}) n={len(it)} seq:',' '.join(f"L{l}{bb[0].upper()}{'o' if ok else 'X'}" for l,bb,ok in s))
