"""Rerun of PoetryClass.ipynb with leakage and coverage fixes.

Parts:
  orig   - the notebook's exact pipeline (direct-child parser, upsample BEFORE split) on its ~14.4K-line subset
  fixed  - full corpus, split first, class balancing on TRAIN only (class weights / uniform prior)
           (a) stratified random 80/20 line split
           (b) 5-fold GroupKFold holding out whole texts
"""
import glob, os, re, sys, time, json, collections
import numpy as np
import xml.etree.ElementTree as ET
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GroupKFold
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score
from sklearn.utils import resample
from lightgbm import LGBMClassifier

DATA = sys.argv[1]
OUT = sys.argv[2]
PARTS = sys.argv[3].split(",") if len(sys.argv) > 3 else ["orig", "fixed"]
NJOBS = 8
SEED = 42
results = {}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def save():
    with open(OUT, "w") as f:
        json.dump(results, f, indent=1)


def parse(mode):
    """mode='direct' reproduces the notebook's './/body/line' (direct children only); 'all' is './/body//line'."""
    rows = []
    for f in sorted(glob.glob(os.path.join(DATA, "*.xml"))):
        body = ET.parse(f).getroot().find("body")
        it = body.findall("line") if mode == "direct" else body.iter("line")
        for l in it:
            t = " ".join(w.text for w in l.iter("word") if w.text).lower()
            t = re.sub(r"[^a-z\s]", "", t)
            if t and l.get("pattern") != "not scanned" and l.get("name"):
                rows.append((t, l.get("meter", ""), os.path.basename(f)))
    return rows


def metrics(y_true, y_pred):
    labs = np.unique(y_true)
    return {
        "acc": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, labels=labs, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, labels=labs, average="weighted", zero_division=0),
        "balanced_acc": balanced_accuracy_score(y_true, y_pred),
    }


def make_model(name):
    if os.environ.get("UNWEIGHTED"):                    # standard models, no class balancing at all
        if name == "NB":
            return MultinomialNB()
        if name == "RF":
            return RandomForestClassifier(n_estimators=100, max_depth=100, n_jobs=NJOBS, random_state=SEED)
        if name == "LGBM":
            return LGBMClassifier(objective="multiclass", n_estimators=100, learning_rate=0.1, num_leaves=31,
                                  n_jobs=NJOBS, verbosity=-1, random_state=SEED)
    if name == "NB":
        return MultinomialNB(fit_prior=False)          # uniform class prior = balancing without duplicating rows
    if name == "RF":
        return RandomForestClassifier(n_estimators=100, max_depth=100, class_weight="balanced_subsample",
                                      n_jobs=NJOBS, random_state=SEED)
    if name == "LGBM":
        return LGBMClassifier(objective="multiclass", n_estimators=100, learning_rate=0.1, num_leaves=31,
                              class_weight="balanced", n_jobs=NJOBS, verbosity=-1, random_state=SEED)
    raise ValueError(name)


def fit_eval(name, Xtr_txt, ytr, Xte_txt, yte, min_df=2, balanced=True):
    vec = CountVectorizer(min_df=min_df)
    Xtr = vec.fit_transform(Xtr_txt).astype(np.float32)   # vocabulary learned on TRAIN only
    Xte = vec.transform(Xte_txt).astype(np.float32)
    m = make_model(name)
    if not balanced and name == "NB":
        m = MultinomialNB()
    t = time.time()
    m.fit(Xtr, ytr)
    pred = m.predict(Xte)
    out = metrics(yte, pred)
    out["vocab"] = Xtr.shape[1]
    out["secs"] = round(time.time() - t, 1)
    return out


def majority_baseline(ytr, yte):
    maj = collections.Counter(ytr).most_common(1)[0][0]
    return metrics(yte, np.full(len(yte), maj))


# ---------------------------------------------------------------- Part 1: original protocol on the notebook's subset
if "orig" in PARTS:
    log("== PART orig: notebook protocol (direct-child parser, upsample-then-split)")
    rows = parse("direct")
    texts = np.array([r[0] for r in rows])
    metr = [r[1] for r in rows]
    enc = {c: i for i, c in enumerate(dict.fromkeys(metr))}
    y = np.array([enc[m] for m in metr])
    vc = collections.Counter(y.tolist())
    maj_cls, maj_n = vc.most_common(1)[0]
    idx_all = [np.arange(len(y))]
    for cls, n in vc.items():
        if n < maj_n:
            idx_cls = np.where(y == cls)[0]
            idx_all.append(resample(idx_cls, replace=True, n_samples=maj_n, random_state=SEED))
    up = np.concatenate(idx_all)
    tx, ty = texts[up], y[up]
    tr_t, te_t, tr_y, te_y = train_test_split(tx, ty, test_size=0.2, random_state=SEED)
    trset = set(tr_t.tolist())
    overlap = float(np.mean([t in trset for t in te_t.tolist()]))
    info = {"rows_original": len(y), "rows_after_upsample": int(len(up)), "classes": len(vc),
            "files": len({r[2] for r in rows}), "test_lines_whose_exact_text_is_also_in_train": overlap,
            "majority_share_original": maj_n / len(y)}
    log(info)
    results["orig_info"] = info
    results["orig"] = {}
    for name in ["NB", "RF"]:
        res = fit_eval(name, tr_t, tr_y, te_t, te_y, min_df=1, balanced=False)  # notebook: plain models, CountVectorizer defaults
        results["orig"][name] = res
        log("orig", name, res)
        save()
    # same subset, done correctly: split FIRST, no duplicated rows, class-balanced training
    yy = np.array(metr)
    tr_t, te_t, tr_y, te_y = train_test_split(texts, yy, test_size=0.2, random_state=SEED,
                                              stratify=np.where(np.array([collections.Counter(metr)[m] for m in metr]) >= 2, yy, "RARE"))
    results["orig_subset_fixed"] = {"majority_baseline": majority_baseline(tr_y, te_y)}
    for name in ["NB", "RF"]:
        res = fit_eval(name, tr_t, tr_y, te_t, te_y, min_df=1)
        results["orig_subset_fixed"][name] = res
        log("orig_subset_fixed", name, res)
        save()

# ---------------------------------------------------------------- Part 2: fixed protocol on the full corpus
if "fixed" in PARTS:
    log("== PART fixed: full corpus")
    rows = [r for r in parse("all") if r[1] != ""]      # drop the 11 lines with an empty meter label
    texts = np.array([r[0] for r in rows])
    y = np.array([r[1] for r in rows])
    groups = np.array([r[2] for r in rows])
    cc = collections.Counter(y.tolist())
    info = {"rows": len(rows), "classes": len(cc), "texts": len(set(groups.tolist())),
            "majority_share": cc.most_common(1)[0][1] / len(rows),
            "class_counts": cc.most_common()}
    results["fixed_info"] = info
    log({k: v for k, v in info.items() if k != "class_counts"})
    save()

    # (a) stratified random 80/20 line split
    tr_t, te_t, tr_y, te_y = train_test_split(texts, y, test_size=0.2, random_state=SEED, stratify=y)
    results["random_split"] = {"majority_baseline": majority_baseline(tr_y, te_y), "n_test": len(te_y)}
    log("random_split majority", results["random_split"]["majority_baseline"])
    for name in ["NB", "RF", "LGBM"]:
        res = fit_eval(name, tr_t, tr_y, te_t, te_y)
        results["random_split"][name] = res
        log("random_split", name, res)
        save()

    # (b) hold out whole texts: 5-fold GroupKFold
    gkf = GroupKFold(n_splits=5)
    folds = list(gkf.split(texts, y, groups))
    results["heldout_texts"] = {"folds": []}
    per_model = collections.defaultdict(list)
    base = []
    for k, (tr, te) in enumerate(folds):
        base.append(majority_baseline(y[tr], y[te]))
        results["heldout_texts"]["folds"].append({"n_test": int(len(te)), "test_texts": sorted(set(groups[te].tolist()))})
        for name in ["NB", "RF", "LGBM"]:
            res = fit_eval(name, texts[tr], y[tr], texts[te], y[te])
            per_model[name].append(res)
            log(f"fold{k}", name, res)
        results["heldout_texts"]["majority_baseline"] = base
        results["heldout_texts"].update({n: v for n, v in per_model.items()})
        save()
    summ = {}
    for name, lst in list(per_model.items()) + [("majority_baseline", base)]:
        summ[name] = {m: (float(np.mean([r[m] for r in lst])), float(np.std([r[m] for r in lst])))
                      for m in ["acc", "macro_f1", "weighted_f1", "balanced_acc"]}
    results["heldout_texts"]["summary"] = summ
    save()
log("DONE")
