"""Pack the R4 outputs for upload (paper Phase 5, D37). Run on the PC in the thesis_xai folder after the
R4 notebook has finished:   python pack_r4_outputs.py            (or: python pack_r4_outputs.py PATH_TO_paper_r4_outputs)
Writes to r4_upload/:
  r4_common.zip      indices/ (all split index files + MANIFEST), r4_models_summary.csv, r4_baselines_summary.csv,
                     env_log_r4_*.json, and the notebook if it is in the folder
  r4_split<k>.zip    everything in split<k>/ (cv_results, oof, test predictions, summary.json, done.json, baselines)
                     for k = 1..5; a split whose zip would exceed MAX_MB is written as two parts (_a, _b)
  SHA256SUMS_r4.txt  SHA-256 of every packed file (path inside the zip) and of every zip
The pipelines/ folder is never packed (keep it on the PC for R5-R6). Before packing, every file listed in a
done.json is checked against its recorded hash, and the run stops if any (split, model) block is missing.
Zips use fixed timestamps, so packing the same outputs twice gives identical zips."""
import hashlib, json, os, sys, zipfile

SRC = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.abspath("."), "paper_r4_outputs")
DST = os.path.join(os.path.dirname(os.path.abspath(SRC)), "r4_upload")
MAX_MB = 30
MODELS = ["LR", "XGB", "LGBM", "MLP", "RF", "CAT"]
STAMP = (2026, 1, 1, 0, 0, 0)


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def rel_files(top):
    out = []
    for root, _, files in os.walk(os.path.join(SRC, top)):
        out += [os.path.relpath(os.path.join(root, f), SRC).replace(os.sep, "/") for f in files if not f.endswith(".tmp")]
    return sorted(out)


def write_zip(name, rels):
    path = os.path.join(DST, name)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for r in rels:
            info = zipfile.ZipInfo(r, date_time=STAMP)
            info.compress_type, info.external_attr = zipfile.ZIP_DEFLATED, 0o644 << 16
            with open(os.path.join(SRC, r), "rb") as f:
                z.writestr(info, f.read(), compresslevel=9)
    with zipfile.ZipFile(path) as z:                      # read-back test
        assert z.testzip() is None and sorted(z.namelist()) == sorted(rels)
    return path


if not os.path.isdir(SRC):
    raise SystemExit(f"{SRC} not found")
splits = sorted(int(d[5:]) for d in os.listdir(SRC) if d.startswith("split") and d[5:].isdigit())
missing, bad = [], []
for k in splits:
    blocks = ["reproduction", "baselines"] if k == 1 else MODELS + ["baselines"]
    for b in blocks:
        dj = os.path.join(SRC, f"split{k}", b, "done.json")
        if not os.path.exists(dj):
            missing.append(f"split{k}/{b}")
            continue
        for r, h in json.load(open(dj, encoding="utf-8"))["files"].items():
            if sha(os.path.join(SRC, r)) != h:
                bad.append(r)
if missing or bad:
    raise SystemExit(f"Not packed. Unfinished blocks: {missing}; files that differ from done.json: {bad}")
os.makedirs(DST, exist_ok=True)
for old in os.listdir(DST):                                # remove zips from an earlier packing
    if old.startswith("r4_") and old.endswith(".zip") or old == "SHA256SUMS_r4.txt":
        os.remove(os.path.join(DST, old))
sums, zips = [], []
common = rel_files("indices") + sorted(f for f in os.listdir(SRC) if os.path.isfile(os.path.join(SRC, f)))
nb = os.path.join(os.path.dirname(os.path.abspath(SRC)), "paper_r4_splits.ipynb")
zips.append(write_zip("r4_common.zip", common))
if os.path.exists(nb):                                     # executed notebook, added at the top level of the zip
    with zipfile.ZipFile(zips[-1], "a", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        info = zipfile.ZipInfo("paper_r4_splits.ipynb", date_time=STAMP); info.compress_type = zipfile.ZIP_DEFLATED
        z.writestr(info, open(nb, "rb").read())
    sums.append((sha(nb), "r4_common.zip:paper_r4_splits.ipynb"))
sums += [(sha(os.path.join(SRC, r)), f"r4_common.zip:{r}") for r in common]
for k in splits:
    rels = rel_files(f"split{k}")
    name = f"r4_split{k}.zip"
    p = write_zip(name, rels)
    if os.path.getsize(p) > MAX_MB * 2 ** 20:              # too large for one upload: two parts by model
        os.remove(p)
        half = MODELS[:3] + ["baselines"]
        parts = {"a": [r for r in rels if r.split("/")[1] in half], "b": [r for r in rels if r.split("/")[1] not in half]}
        for tag, rr in parts.items():
            zips.append(write_zip(f"r4_split{k}_{tag}.zip", rr))
            sums += [(sha(os.path.join(SRC, r)), f"r4_split{k}_{tag}.zip:{r}") for r in rr]
    else:
        zips.append(p)
        sums += [(sha(os.path.join(SRC, r)), f"{name}:{r}") for r in rels]
with open(os.path.join(DST, "SHA256SUMS_r4.txt"), "w", encoding="utf-8", newline="\n") as f:
    for z in zips:
        f.write(f"{sha(z)}  {os.path.basename(z)}\n")
    for h, r in sums:
        f.write(f"{h}  {r}\n")
print(f"Packed {len(sums)} files into {len(zips)} zips in {DST}:")
for z in zips:
    print(f"  {os.path.basename(z):22s} {os.path.getsize(z) / 2 ** 20:6.1f} MB  {sha(z)}")
print("Upload every zip and SHA256SUMS_r4.txt. The pipelines/ folder was not packed; keep it on the PC.")
