# Probe fit + decode on the DK1 latent exports, two targets:
#   abs      = the stored action (absolute EE pose target, ~ state 5 frames ahead)
#   delta_h5 = state[t+5] - state[t] within the episode (realized EE motion over the LAM horizon H=5)
# Ridge(alpha=1) and a [512,256] ReLU MLP (torch, CPU; see TorchMLP) on StandardScaler'd latents, fitted on robot_3cam rows.
# Writes probe_realworld_local.csv (per-dim R^2 on a 20%-of-episodes holdout) and
# decoded_<lam>.parquet with gt_*/ridge_*/mlp_* (abs) and gt_d*/ridge_d*/mlp_d* (delta) per frame.
import os, time, pandas as pd, numpy as np, pyarrow.dataset as ds
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score
H=5; DIMS=["x","y","z","wx","wy","wz","grip"]
src=pd.read_parquet("exports/sharedlam2/meta/source_episodes.parquet")[["episode_index","task","source_kind"]]
import torch
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS","8")))
class TorchMLP:
    """[512,256] ReLU MLP, Adam 1e-3, batch 2048, <=80 epochs, early stopping on a 10% val split (patience 8).
    CPU replacement for sklearn's MLPRegressor, which took >90 min per LAM on the loaded workstation."""
    def __init__(self,seed=0): self.seed=seed
    def fit(self,X,Y):
        torch.manual_seed(self.seed); rng=np.random.RandomState(self.seed); n=len(X); perm=rng.permutation(n); nv=n//10
        Xt=torch.tensor(X,dtype=torch.float32); Yt=torch.tensor(Y,dtype=torch.float32)
        self.ym=Yt[perm[nv:]].mean(0); self.ys=Yt[perm[nv:]].std(0)+1e-6; Yn=(Yt-self.ym)/self.ys
        tr,va=perm[nv:],perm[:nv]
        self.net=torch.nn.Sequential(torch.nn.Linear(X.shape[1],512),torch.nn.ReLU(),torch.nn.Linear(512,256),torch.nn.ReLU(),torch.nn.Linear(256,Y.shape[1]))
        opt=torch.optim.Adam(self.net.parameters(),1e-3); best=1e9; bad=0; state=None
        for ep in range(80):
            self.net.train(); p=torch.tensor(rng.permutation(tr))
            for i in range(0,len(p),2048):
                b=p[i:i+2048]; opt.zero_grad(); loss=torch.nn.functional.mse_loss(self.net(Xt[b]),Yn[b]); loss.backward(); opt.step()
            self.net.eval()
            with torch.no_grad(): vl=torch.nn.functional.mse_loss(self.net(Xt[va]),Yn[va]).item()
            if vl<best-1e-5: best=vl; bad=0; state={k:v.clone() for k,v in self.net.state_dict().items()}
            else:
                bad+=1
                if bad>=8: break
        self.net.load_state_dict(state); self.epochs=ep+1; return self
    def predict(self,X):
        self.net.eval()
        with torch.no_grad(): return (self.net(torch.tensor(X,dtype=torch.float32))*self.ys+self.ym).numpy()
MK={"ridge":lambda: Ridge(alpha=1.0),"mlp":lambda: TorchMLP(0)}
PROBES=os.environ.get("PROBES","ridge,mlp").split(",")
rows=[]
for m,method in [("sharedlam2","ours_multi"),("sharedlam3side","ours_single")]:
    t=ds.dataset(f"exports/{m}/data",format="parquet").to_table(columns=["episode_index","frame_index","action","observation.state","latent_labels.continuous_vector_latents","latent_labels.valid"]).to_pandas().merge(src,on="episode_index").sort_values(["episode_index","frame_index"]).reset_index(drop=True)
    Z=np.stack([np.concatenate([np.asarray(x,dtype=np.float32).ravel() for x in a]) for a in t["latent_labels.continuous_vector_latents"].values])
    A=np.stack(t.action.values); S=np.stack(t["observation.state"].values); v=t["latent_labels.valid"].values.astype(bool)
    eps=t.episode_index.values; same=np.roll(eps,-H)==eps; D=np.roll(S,-H,axis=0)-S
    targets={"abs":(A,v),"delta_h5":(D,v&same)}
    rng=np.random.RandomState(42); reps=np.unique(eps[(t.source_kind.values=="robot_3cam")&v]); te_eps=rng.choice(reps,int(0.2*len(reps)),replace=False)
    df=pd.DataFrame(dict(episode_index=eps,frame_index=t.frame_index.values,valid=v,source_kind=t.source_kind.values,task=t.task.values))
    for tname,(Y,ok) in targets.items():
        rob=(t.source_kind.values=="robot_3cam")&ok; te=rob&np.isin(eps,te_eps); tr=rob&~np.isin(eps,te_eps)
        sc=StandardScaler().fit(Z[tr]); pre="" if tname=="abs" else "d"
        for d,n in enumerate(DIMS): df[f"gt_{pre}{n}"]=np.where(ok,Y[:,d],np.nan)
        for pname in PROBES:
            t0=time.time(); mdl=MK[pname]().fit(sc.transform(Z[tr]),Y[tr]); pr=mdl.predict(sc.transform(Z[te]))
            r2=[r2_score(Y[te][:,d],pr[:,d]) for d in range(7)]
            print(f"{method:12s} {tname:9s} {pname:5s} {np.round(r2,3)} mean {np.mean(r2):.3f} xyz {np.mean(r2[:3]):.3f} ({time.time()-t0:.0f}s)",flush=True)
            for d,n in enumerate(DIMS): rows.append(dict(method=method,target=tname,probe=pname,action_dim=d,r2=r2[d],n_train=int(tr.sum()),n_test=int(te.sum())))
            rows.append(dict(method=method,target=tname,probe=pname,action_dim=-1,r2=float(np.mean(r2)),n_train=int(tr.sum()),n_test=int(te.sum())))
            full=MK[pname]().fit(sc.transform(Z[rob]),Y[rob]); dec=np.full_like(Y,np.nan); dec[ok]=full.predict(sc.transform(Z[ok]))
            for d,n in enumerate(DIMS): df[f"{pname}_{pre}{n}"]=dec[:,d]
    df.to_parquet(f"decoded_{m}.parquet")
    pd.DataFrame(rows).to_csv("probe_realworld_local.csv",index=False)
print("saved")
