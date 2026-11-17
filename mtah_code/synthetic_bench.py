import numpy as np
rng = np.random.default_rng(1)
FPS=25
def make_match(n_poss=40, dur=(5,25), mu=3.0, sigma=6.0, miss_p=0.41):
    labs=[];  side=0
    for i in range(n_poss):
        d=int(rng.uniform(*dur)*FPS); labs += [side]*d; side=1-side
    truth=np.array(labs)
    e=np.where(truth==0, mu, -mu)+rng.normal(0,sigma,len(truth))   # >0 home nearer
    # missing ball in bursts
    vis=np.ones(len(truth),bool); i=0
    while i<len(vis):
        if rng.random()<miss_p/50:  # burst starts
            L=int(rng.exponential(50)); vis[i:i+L]=False; i+=L
        else: i+=1
    return truth,e,vis
def spans(lab):
    out=[];s=0
    for i in range(1,len(lab)+1):
        if i==len(lab) or lab[i]!=lab[s]: out.append((lab[s],s,i)); s=i
    return out
def iou(a,b):
    inter=max(0,min(a[2],b[2])-max(a[1],b[1])); return inter/((a[2]-a[1])+(b[2]-b[1])-inter)
def spell_f1(pred,truth):
    P=spans(pred);T=spans(truth);f=[]
    for th in (.3,.4,.5,.6,.7):
        used=set();tp=0
        for p in P:
            best=None;bi=0
            for j,t in enumerate(T):
                if j in used or t[0]!=p[0]: continue
                v=iou(p,t)
                if v>bi: bi,best=v,j
            if best is not None and bi>=th: used.add(best);tp+=1
        pr=tp/len(P);rc=tp/len(T);f.append(0 if tp==0 else 2*pr*rc/(pr+rc))
    return np.mean(f)
def window_vote(e,vis,w):
    code=np.where(vis, (e>0).astype(float), np.nan)
    h=np.nan_to_num(np.where(code==1,1,0)).astype(float); a=np.where(code==0,1,0).astype(float)
    ch=np.concatenate(([0],np.cumsum(h)));ca=np.concatenate(([0],np.cumsum(a)))
    idx=np.arange(len(e));lo=np.maximum(0,idx-w);hi=np.minimum(len(e),idx+w+1)
    H=ch[hi]-ch[lo];A=ca[hi]-ca[lo]
    return np.where(H>=A,0,1)
def viterbi2(e,vis,s=3.0,stay=0.9995):
    # log-lik of evidence: logistic evidence; missing -> 0
    llr=np.where(vis, e/s, 0.0)           # log-likelihood ratio home vs away
    ll=np.stack([ llr/2, -llr/2],1)
    T=len(e); lt=np.log(np.array([[stay,1-stay],[1-stay,stay]]))
    dp=np.zeros((T,2));bp=np.zeros((T,2),int);dp[0]=ll[0]
    for t in range(1,T):
        c0=dp[t-1][:,None]+lt
        bp[t]=c0.argmax(0);dp[t]=c0.max(0)+ll[t]
    path=[int(dp[-1].argmax())]
    for t in range(T-1,0,-1): path.append(int(bp[t,path[-1]]))
    return np.array(path[::-1])
res={}
for seed in range(5):
    rng=np.random.default_rng(seed)
    truth,e,vis=make_match()
    r={'raw':spell_f1(window_vote(e,vis,0),truth)}
    for w in (25,50,125,250): r[f'win{w}']=spell_f1(window_vote(e,vis,w),truth)
    for stay in (0.99,0.999,0.9995,0.9999): r[f'vit{stay}']=spell_f1(viterbi2(e,vis,3.0,stay),truth)
    for k,v in r.items(): res.setdefault(k,[]).append(v)
for k,v in res.items(): print(k, round(np.mean(v),3))
