import json, math, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
R = json.load(open("results/results.json"))
S = R["stage2"]
def pick(f): return next(x for x in S if f(x))
rows = [("original (2 clients)", R["original"]), ("retrain (Mac only)", R["retrain"]),
        ("subtract updates", R["subtract"]),
        ("GA only (stop loss≥ln10)", pick(lambda x: x["kind"]=="ga_stop" and abs(x["target"]-math.log(10))<1e-6)),
        ("GA + 1 repair round", pick(lambda x: x["kind"]=="ga_repair" and abs(x["target"]-math.log(10))<1e-6 and x["repair_rounds"]==1)),
        ("FedEraser-mix α=0.25", pick(lambda x: x["kind"]=="federaser_mix" and x["alpha"]==0.25))]
names=[n for n,_ in rows]
plt.rcParams.update({"font.family":"AppleGothic","axes.unicode_minus":False})
fig,ax=plt.subplots(1,3,figsize=(15,4.8))
for a,(k,t) in zip(ax,[("global_test_acc","CIFAR-10 test acc (↑ 유지)"),("forget_train_acc","forget set(desktop data) acc (retrain에 가까울수록 좋음)"),("mia_auc","MIA AUC (0.5 = 멤버 구분 불가)")]):
    v=[r[k] for _,r in rows]; cols=["#888","#2a9d8f","#bbb","#e76f51","#f4a261","#264653"]
    a.barh(names,v,color=cols); a.invert_yaxis(); a.set_title(t,fontsize=9)
    for i,x in enumerate(v): a.text(x,i,f" {x:.3f}",va="center",fontsize=8)
    if k!="global_test_acc": a.set_yticklabels([])
    if k=="mia_auc": a.axvline(0.5,ls="--",c="k",lw=.7)
plt.tight_layout(); plt.savefig("compare.png",dpi=140)
fig,ax=plt.subplots(figsize=(9,4.5))
import numpy as np
cl=["plane","car","bird","cat","deer","dog","frog","horse","ship","truck"]
for n,r in rows[:2]+rows[4:]: ax.plot(cl,r["per_class_acc"],marker="o",label=n)
ax.set_title("class-wise test acc (desktop-heavy: cat/frog/horse/ship)"); ax.legend(fontsize=8); plt.tight_layout(); plt.savefig("per_class.png",dpi=140)
print("\n".join(f"{n:28s} test {r['global_test_acc']:.3f} forget {r['forget_train_acc']:.3f} retain_te {r['retain_test_acc']:.3f} MIA {r['mia_auc']:.3f} cost {r['local_epoch_cost']:.1f}ep" for n,r in rows))
