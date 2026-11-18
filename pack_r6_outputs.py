"""Pack the R6 outputs for upload (paper Phase 5, D39). Run on the PC in thesis_xai after the R6 notebook:
    python pack_r6_outputs.py            (or: python pack_r6_outputs.py PATH_TO_paper_r6_outputs [MAX_MB])
Writes r6_upload/:
  r6_common.zip     seeds_r6.json, rf_fallback.json, refits_r6.csv, additivity_r6.csv, timing_r6.csv, gates_r6.csv,
                    r5_regression_check.csv, env log, models/*.done.json (refit records: hash, seeds, test AP), and the
                    executed notebook paper_r6_refits.ipynb (if it is next to the outputs folder)
  r6_split<k>.zip   split<k>/native and split<k>/common: SHAP files and their .done.json; a zip above MAX_MB
                    (default 25) is written as one zip per setting (and, if still too big, per refit kind)
  SHA256SUMS_r6.txt SHA-256 of every zip and of every packed file ("zip:path")
Never packed: _work/ (chunks) and the model files models/*.joblib (keep them on the PC).
Before packing, every SHAP file planned in seeds_r6.json must exist and match its .done.json hash, and every refit must
have its record; otherwise nothing is packed. Zips use fixed timestamps, so packing twice gives identical zips."""
import hashlib, json, os, sys, zipfile

SRC = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.abspath("."), "paper_r6_outputs")
MAX_MB = float(sys.argv[2]) if len(sys.argv) > 2 else 25
DST = os.path.join(os.path.dirname(os.path.abspath(SRC)), "r6_upload")
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
        for arc, src in [(r, os.path.join(SRC, r)) for r in rels] + list((extra or {}).items()):
            info = zipfile.ZipInfo(arc, date_time=STAMP)
            info.compress_type, info.external_attr = zipfile.ZIP_DEFLATED, 0o644 << 16
            z.writestr(info, open(src, "rb").read(), compresslevel=9)
    with zipfile.ZipFile(path) as z:                                   # read-back test
        assert z.testzip() is None and sorted(z.namelist()) == sorted(list(rels) + list(extra or {}))
    return path


if not os.path.isdir(SRC):
    raise SystemExit(f"{SRC} not found")
seeds = json.load(open(os.path.join(SRC, "seeds_r6.json"), encoding="utf-8"))
missing, bad = [], []
for u in seeds["units"]:
    if not os.path.exists(os.path.join(SRC, "models", u["unit"] + ".done.json")):
        missing.append("models/" + u["unit"] + ".done.json")
    for s in u["settings"]:
        stem = u["unit"][len(f"split{u['split']}_"):]
        dj = os.path.join(SRC, f"split{u['split']}", s, f"SHAP_{stem}.done.json")
        if not os.path.exists(dj):
            missing.append(f"split{u['split']}/{s}/SHAP_{stem}"); continue
        rec = json.load(open(dj, encoding="utf-8"))
        if sha(os.path.join(SRC, rec["file"])) != rec["sha256"]:
            bad.append(rec["file"])
for f in ["refits_r6.csv", "additivity_r6.csv", "gates_r6.csv", "r5_regression_check.csv"]:
    if not os.path.exists(os.path.join(SRC, f)):
        missing.append(f)
if missing or bad:
    raise SystemExit(f"Not packed. Missing ({len(missing)}): {missing[:10]}; files that differ from their .done.json: {bad[:10]}")
os.makedirs(DST, exist_ok=True)
for old in os.listdir(DST):
    if (old.startswith("r6_") and old.endswith(".zip")) or old == "SHA256SUMS_r6.txt":
        os.remove(os.path.join(DST, old))
rel = lambda p: os.path.relpath(p, SRC).replace(os.sep, "/")
common = sorted(f for f in os.listdir(SRC) if os.path.isfile(os.path.join(SRC, f)) and not f.endswith(".tmp"))
common += sorted("models/" + f for f in os.listdir(os.path.join(SRC, "models")) if f.endswith(".done.json"))
nb = os.path.join(os.path.dirname(os.path.abspath(SRC)), "paper_r6_refits.ipynb")
extra = {"paper_r6_refits.ipynb": nb} if os.path.exists(nb) else {}
zips = [write_zip("r6_common.zip", common, extra)]
sums = [(sha(os.path.join(SRC, r)), f"r6_common.zip:{r}") for r in common] + [(sha(p), f"r6_common.zip:{a}") for a, p in extra.items()]
splits = sorted({u["split"] for u in seeds["units"]})
for k in splits:
    groups = {}
    for s in ["native", "common"]:
        d = os.path.join(SRC, f"split{k}", s)
        if os.path.isdir(d):
            groups[s] = sorted(rel(os.path.join(d, f)) for f in os.listdir(d) if not f.endswith(".tmp"))
    allr = [r for g in groups.values() for r in g]
    p = write_zip(f"r6_split{k}.zip", allr)
    if os.path.getsize(p) <= MAX_MB * 2 ** 20:
        zips.append(p); sums += [(sha(os.path.join(SRC, r)), f"r6_split{k}.zip:{r}") for r in allr]
        continue
    os.remove(p)
    for s, rr in groups.items():                                       # one zip per setting
        p = write_zip(f"r6_split{k}_{s}.zip", rr)
        if os.path.getsize(p) <= MAX_MB * 2 ** 20:
            zips.append(p); sums += [(sha(os.path.join(SRC, r)), f"r6_split{k}_{s}.zip:{r}") for r in rr]
            continue
        os.remove(p)
        kinds = {}
        for r in rr:                                                   # one zip per refit kind (boot / seed / rash)
            kinds.setdefault(os.path.basename(r).split("_")[1][:4], []).append(r)
        for kind, kr in sorted(kinds.items()):
            zips.append(write_zip(f"r6_split{k}_{s}_{kind}.zip", kr))
            sums += [(sha(os.path.join(SRC, r)), f"r6_split{k}_{s}_{kind}.zip:{r}") for r in kr]
too_big = [z for z in zips if os.path.getsize(z) > MAX_MB * 2 ** 20]
if too_big:
    raise SystemExit(f"zips above {MAX_MB} MB: {too_big}")
with open(os.path.join(DST, "SHA256SUMS_r6.txt"), "w", encoding="utf-8", newline="\n") as f:
    for z in zips:
        f.write(f"{sha(z)}  {os.path.basename(z)}\n")
    for h, r in sums:
        f.write(f"{h}  {r}\n")
print(f"Packed {len(sums)} files into {len(zips)} zips in {DST}:")
for z in zips:
    print(f"  {os.path.basename(z):28s} {os.path.getsize(z) / 2 ** 20:6.2f} MB  {sha(z)}")
print("Upload every zip and SHA256SUMS_r6.txt. _work/ and models/*.joblib were not packed; keep the models on the PC.")
