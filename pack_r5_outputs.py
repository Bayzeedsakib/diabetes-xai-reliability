"""Pack the R5 outputs for upload (paper Phase 5, D37 amendment). Run on the PC in thesis_xai after the R5 notebook:
    python pack_r5_outputs.py            (or: python pack_r5_outputs.py PATH_TO_paper_r5_outputs [MAX_MB])
Writes r5_upload/:
  r5_common.zip     gates, reproduction, regression check, additivity, ALT_XGB top-5, timing, env log, the executed
                    notebook paper_r5_shap.ipynb (if it is next to the outputs folder)
  r5_split<k>.zip   split<k>/native, split<k>/common (and split1/xgbcheck): SHAP files and their .done.json;
                    a zip above MAX_MB (default 25) is written as one zip per setting
  SHA256SUMS_r5.txt SHA-256 of every zip and of every packed file ("zip:path")
Never packed: _work/ (chunks) and models/ (refitted models; keep them on the PC for R6).
Before packing every expected SHAP file must exist and match its .done.json hash; otherwise nothing is packed.
Zips use fixed timestamps, so packing the same outputs twice gives identical zips."""
import hashlib, json, os, sys, zipfile

SRC = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.abspath("."), "paper_r5_outputs")
MAX_MB = float(sys.argv[2]) if len(sys.argv) > 2 else 25
DST = os.path.join(os.path.dirname(os.path.abspath(SRC)), "r5_upload")
MODELS = ["LR", "XGB", "LGBM", "MLP", "RF", "CAT"]
STAMP = (2026, 1, 1, 0, 0, 0)


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def write_zip(name, rels, extra=None):
    path = os.path.join(DST, name)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for r in rels:
            info = zipfile.ZipInfo(r, date_time=STAMP)
            info.compress_type, info.external_attr = zipfile.ZIP_DEFLATED, 0o644 << 16
            z.writestr(info, open(os.path.join(SRC, r), "rb").read(), compresslevel=9)
        for arc, src in (extra or {}).items():
            info = zipfile.ZipInfo(arc, date_time=STAMP)
            info.compress_type, info.external_attr = zipfile.ZIP_DEFLATED, 0o644 << 16
            z.writestr(info, open(src, "rb").read(), compresslevel=9)
    with zipfile.ZipFile(path) as z:                                   # read-back test
        assert z.testzip() is None and sorted(z.namelist()) == sorted(list(rels) + list(extra or {}))
    return path


if not os.path.isdir(SRC):
    raise SystemExit(f"{SRC} not found")
splits = sorted(int(d[5:]) for d in os.listdir(SRC) if d.startswith("split") and d[5:].isdigit())
expected = []
for k in splits:
    expected += [f"split{k}/native/SHAP_{m}" for m in MODELS + ["ALT_XGB"]] + [f"split{k}/common/SHAP_{m}" for m in MODELS]
    if k == 1:
        expected += ["split1/xgbcheck/SHAP_XGB_tree100", "split1/xgbcheck/SHAP_XGB_perm_logodds"]
missing, bad = [], []
for e in expected:
    dj = os.path.join(SRC, e + ".done.json")
    if not os.path.exists(dj):
        missing.append(e); continue
    rec = json.load(open(dj, encoding="utf-8"))
    if sha(os.path.join(SRC, rec["file"])) != rec["sha256"]:
        bad.append(rec["file"])
for k in splits:
    if not os.path.exists(os.path.join(SRC, f"split{k}", "reproduction", "done.json")):
        missing.append(f"split{k}/reproduction")
for f in ["additivity.csv", "alt_xgb_top5.csv", "gates.csv", "reproduction_all.csv"]:
    if not os.path.exists(os.path.join(SRC, f)):
        missing.append(f)
if missing or bad:
    raise SystemExit(f"Not packed. Missing: {missing}; files that differ from their .done.json: {bad}")
os.makedirs(DST, exist_ok=True)
for old in os.listdir(DST):
    if (old.startswith("r5_") and old.endswith(".zip")) or old == "SHA256SUMS_r5.txt":
        os.remove(os.path.join(DST, old))
rel = lambda p: os.path.relpath(p, SRC).replace(os.sep, "/")
common = sorted(f for f in os.listdir(SRC) if os.path.isfile(os.path.join(SRC, f)))
for k in splits:
    for root, _, files in os.walk(os.path.join(SRC, f"split{k}")):
        if os.path.basename(root) not in ("native", "common", "xgbcheck"):
            common += [rel(os.path.join(root, f)) for f in sorted(files) if not f.endswith(".tmp")]
nb = os.path.join(os.path.dirname(os.path.abspath(SRC)), "paper_r5_shap.ipynb")
extra = {"paper_r5_shap.ipynb": nb} if os.path.exists(nb) else {}
zips, sums = [write_zip("r5_common.zip", common, extra)], [(sha(os.path.join(SRC, r)), f"r5_common.zip:{r}") for r in common]
sums += [(sha(p), f"r5_common.zip:{a}") for a, p in extra.items()]
for k in splits:
    groups = {}
    for s in ["native", "common", "xgbcheck"]:
        d = os.path.join(SRC, f"split{k}", s)
        if os.path.isdir(d):
            groups[s] = sorted(rel(os.path.join(d, f)) for f in os.listdir(d) if not f.endswith(".tmp"))
    allr = [r for g in groups.values() for r in g]
    p = write_zip(f"r5_split{k}.zip", allr)
    if os.path.getsize(p) > MAX_MB * 2 ** 20:                          # one zip per setting
        os.remove(p)
        for s, rr in groups.items():
            zips.append(write_zip(f"r5_split{k}_{s}.zip", rr)); sums += [(sha(os.path.join(SRC, r)), f"r5_split{k}_{s}.zip:{r}") for r in rr]
    else:
        zips.append(p); sums += [(sha(os.path.join(SRC, r)), f"r5_split{k}.zip:{r}") for r in allr]
too_big = [z for z in zips if os.path.getsize(z) > MAX_MB * 2 ** 20]
if too_big:
    raise SystemExit(f"zips above {MAX_MB} MB even per setting: {too_big}")
with open(os.path.join(DST, "SHA256SUMS_r5.txt"), "w", encoding="utf-8", newline="\n") as f:
    for z in zips:
        f.write(f"{sha(z)}  {os.path.basename(z)}\n")
    for h, r in sums:
        f.write(f"{h}  {r}\n")
print(f"Packed {len(sums)} files into {len(zips)} zips in {DST}:")
for z in zips:
    print(f"  {os.path.basename(z):24s} {os.path.getsize(z) / 2 ** 20:6.2f} MB  {sha(z)}")
print("Upload every zip and SHA256SUMS_r5.txt. _work/ and models/ were not packed; keep models/ on the PC for R6.")
