"""Create a self-contained ZIP from committed files, excluding local run outputs."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import zipfile

REPO = Path(__file__).resolve().parents[1]
PACKAGE = "deliverables/FKGE_BRSET_Tan"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", default="HEAD")
    args = parser.parse_args()
    revision = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", args.revision], text=True).strip()
    raw = subprocess.check_output(["git", "-C", str(REPO), "archive", "--format=zip", revision, PACKAGE])
    source = zipfile.ZipFile(io.BytesIO(raw))
    contents = {item.filename.removeprefix("deliverables/"): source.read(item)
                for item in source.infolist() if not item.is_dir()}
    for name in contents:
        if any(part in {"private", "outputs", "__pycache__"} for part in Path(name).parts):
            raise RuntimeError(f"Run artifacts must not be released: {name}")
    manifest = json.loads(contents["FKGE_BRSET_Tan/data/release_manifest.json"])
    for name, expected in manifest["files_sha256"].items():
        actual = hashlib.sha256(contents[f"FKGE_BRSET_Tan/data/{name}"]).hexdigest()
        if actual != expected:
            raise RuntimeError(f"Dataset checksum differs: {name}")
    provenance = json.loads(contents["FKGE_BRSET_Tan/SOURCE_PROVENANCE.json"])
    for name, expected in provenance["reference_files"].items():
        if hashlib.sha256(contents[f"FKGE_BRSET_Tan/reference/{name}"]).hexdigest() != expected:
            raise RuntimeError(f"Original reference file differs: {name}")
    release = {"code_commit": revision, "images": 16266, "patients": 8524, "outer_folds": 5,
               "sha256": {name.removeprefix("FKGE_BRSET_Tan/"): hashlib.sha256(value).hexdigest()
                          for name, value in contents.items()}}
    contents["FKGE_BRSET_Tan/RELEASE.json"] = (json.dumps(release, indent=2) + "\n").encode()
    output = REPO / "deliverables/FKGE_BRSET_Tan.zip"
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name, value in sorted(contents.items()):
            item = zipfile.ZipInfo(name, (2026, 11, 1, 0, 0, 0))
            item.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(item, value)
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    output.with_suffix(".zip.sha256").write_text(f"{digest}  {output.name}\n", encoding="utf-8")
    print(json.dumps({"file": str(output), "bytes": output.stat().st_size,
                      "code_commit": revision, "sha256": digest}, indent=2))


if __name__ == "__main__":
    main()
