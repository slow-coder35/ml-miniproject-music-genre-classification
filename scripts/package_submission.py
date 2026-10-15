"""Package source, evidence, documents, and only the selected demo model."""

import hashlib
import json
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "deliverables"
selection = json.loads((ROOT / "results/demo_model.json").read_text())
audit = ROOT / "results/audit"
audit.mkdir(exist_ok=True)
for name in (
    "inventory.json",
    "inventory.csv",
    "splits.json",
    "split_meta.json",
    "windows.json",
    "feature_meta.json",
):
    shutil.copy2(ROOT / "data/cache" / name, audit / name)
shutil.copy2(ROOT / "data/download.json", audit / "download.json")
selected = ROOT / selection["model_path"]
files = []
for name in (
    "README.md",
    "pyproject.toml",
    "requirements-dev.txt",
    ".gitignore",
    "requirements.txt",
    "requirements-gpu.txt",
    "environment-lock.txt",
    "team.json",
    "app.py",
):
    files.append(ROOT / name)
for folder in ("src", "scripts", "tests", "docs", "results"):
    for path in (ROOT / folder).rglob("*"):
        if not path.is_file() or "__pycache__" in path.parts or path.suffix == ".pyc":
            continue
        if "models" in path.parts and path != selected:
            continue
        files.append(path)
for pattern in ("*.pdf", "*.pptx"):
    files.extend(OUT.glob(pattern))
checks = {
    str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in files
}
(OUT / "SHA256SUMS.json").write_text(json.dumps(checks, indent=2) + "\n")
with zipfile.ZipFile(
    OUT / "music_genre_submission.zip",
    "w",
    compression=zipfile.ZIP_DEFLATED,
    compresslevel=6,
) as archive:
    for path in files:
        archive.write(path, "music_genre_classification/" + str(path.relative_to(ROOT)))
    archive.write(OUT / "SHA256SUMS.json", "music_genre_classification/SHA256SUMS.json")
print(OUT / "music_genre_submission.zip")
