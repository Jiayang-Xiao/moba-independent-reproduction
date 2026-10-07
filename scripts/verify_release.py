"""Validate exactly the packaged code/config/provenance files before executing them."""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    record = json.loads((ROOT/"release_manifest.json").read_text(encoding="utf-8"))
    for name, declared in record["files"].items():
        path = (ROOT/name).resolve()
        if ROOT not in path.parents: raise RuntimeError("Unsafe release path")
        data = path.read_bytes()
        if len(data) != declared["bytes"] or hashlib.sha256(data).hexdigest() != declared["sha256"]:
            raise RuntimeError("Changed release file: " + name)
        if path.suffix == ".py": ast.parse(data.decode("utf-8"), filename=name)
    config = json.loads((ROOT/"config.json").read_text())
    if config["identity"] != record["identity"]: raise RuntimeError("Release/config identity mismatch")
    if not config["constraints"]["source_normal_only"] or config["constraints"]["target_training"]:
        raise RuntimeError("Invalid zero-shot contract")
    print("Verified", len(record["files"]), "release files, including fixed config and upstream identities", flush=True)


if __name__ == "__main__": main()
