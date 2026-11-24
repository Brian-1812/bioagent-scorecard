"""Task records and task loading.

A task is one multiple-choice question from a benchmark subset.
The harness stores tasks as JSONL files under data/ (git ignores data/).
The `fetch` command downloads the parquet file of a LAB-Bench subset at the
revision pinned in pins.json, checks two SHA-256 values (parquet and JSONL), and
writes a meta file with the revision and both values. A run records that meta in its manifest, so a reader can check
that two runs used the same questions.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import time
import urllib.request
from pathlib import Path

HF_DATASET = "futurehouse/lab-bench"

LAB_BENCH_SUBSETS = (
    "CloningScenarios", "DbQA", "FigQA", "LitQA2",
    "ProtocolQA", "SeqQA", "SuppQA", "TableQA",
)


@dataclasses.dataclass(frozen=True)
class Task:
    id: str
    benchmark: str
    subset: str
    subtask: str
    question: str
    ideal: str
    distractors: tuple[str, ...]
    canary: str = ""

    @classmethod
    def from_dict(cls, d: dict) -> "Task":
        return cls(
            id=str(d["id"]),
            benchmark=d.get("benchmark", "lab-bench"),
            subset=d.get("subset", ""),
            subtask=d.get("subtask") or d.get("subset", ""),
            question=d["question"],
            ideal=d["ideal"],
            distractors=tuple(d.get("distractors") or ()),
            canary=d.get("canary") or "",
        )

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d["distractors"] = list(self.distractors)
        return d


def load_jsonl(path: str | Path) -> list[Task]:
    tasks = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                tasks.append(Task.from_dict(json.loads(line)))
    ids = [t.id for t in tasks]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{path}: duplicate task ids")
    return tasks


def load_meta(path: str | Path) -> dict:
    meta_path = Path(str(path) + ".meta.json")
    meta = {"source": "local file, no meta"}
    p = Path(path)
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    elif p.parent.name == "lab-bench" and p.stem in LAB_BENCH_SUBSETS:
        # No meta file, but the path says what it is: the pin still applies.
        meta = {"source": f"hf://datasets/{HF_DATASET} (assumed from the path; no meta file)",
                "config": p.stem}
    meta["sha256_now"] = sha256_file(path)
    if "sha256" in meta and meta["sha256"] != meta["sha256_now"]:
        raise ValueError(f"{path}: content changed after fetch (sha256 mismatch)")
    return meta


PINS_FILE = Path(__file__).with_name("pins.json")


def pin_key(meta: dict) -> str | None:
    """'lab-bench/SeqQA' for a file that `fetch` wrote, else None."""
    if str(meta.get("source", "")).startswith(f"hf://datasets/{HF_DATASET}") and meta.get("config"):
        return f"lab-bench/{meta['config']}"
    return None


def pin_status(meta: dict) -> str:
    """'match', 'mismatch' or 'unpinned' (no pin for this file or not a fetched file)."""
    key = pin_key(meta)
    pins = json.loads(PINS_FILE.read_text(encoding="utf-8"))
    if key is None or key not in pins:
        return "unpinned"
    return "match" if pins[key]["sha256"] == meta.get("sha256_now") else "mismatch"


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(url: str, retries: int = 3) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "bioagent-scorecard/0.1"})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return r.read()
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError("unreachable")


def _read_parquet_rows(path: Path) -> list[dict]:
    """Read all rows of a parquet file. Only `fetch` needs this; runs do not."""
    try:
        import pyarrow.parquet as pq
    except ImportError as e:
        raise RuntimeError("fetch reads the Hugging Face parquet file and needs pyarrow "
                           "(pip install pyarrow). Runs do not need it.") from e
    return pq.ParquetFile(str(path)).read().to_pylist()


def parquet_url(pin: dict) -> str:
    """The file URL at the pinned commit (the git tree, not a cache or a branch)."""
    return (f"https://huggingface.co/datasets/{pin['hf_dataset']}/resolve/"
            f"{pin['hf_revision']}/{pin['parquet_path']}")


def fetch_lab_bench(subset: str, out_dir: str | Path, *, pins: dict | None = None,
                    download=_download, read_rows=_read_parquet_rows) -> Path:
    """Download one LAB-Bench subset to <out_dir>/lab-bench/<subset>.jsonl.

    Stale-data guard. An earlier version read the datasets-server /rows
    cache. That cache served the content of an older commit (2050d471) and the
    meta file recorded the newest commit (5c77cec6), so the meta was false.
    Now the fetch:
    1. reads the revision, the parquet path and two SHA-256 values from pins.json;
    2. downloads the parquet file of that revision from the git tree;
    3. stops with an error if the parquet SHA-256 is not the pinned value;
    4. converts the rows to JSONL and stops if that SHA-256 is not the pinned value.
    A subset with no pin stops with an error. The old good files stay unchanged
    when a check fails. This makes network calls to Hugging Face, no model calls.
    """
    if subset not in LAB_BENCH_SUBSETS:
        raise ValueError(f"unknown subset {subset!r}; use one of {LAB_BENCH_SUBSETS}")
    key = f"lab-bench/{subset}"
    pins = pins if pins is not None else json.loads(PINS_FILE.read_text(encoding="utf-8"))
    pin = pins.get(key)
    need = ("hf_dataset", "hf_revision", "parquet_path", "parquet_sha256", "sha256")
    if not pin or any(not pin.get(k) for k in need):
        raise RuntimeError(f"no complete pin for {key} in pins.json (need {', '.join(need)}); "
                           "add the revision and the two SHA-256 values before you fetch")
    out = Path(out_dir) / "lab-bench" / f"{subset}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    url = parquet_url(pin)
    raw = download(url)
    pq_path = out.with_suffix(".parquet")
    pq_tmp = out.with_suffix(".parquet.tmp")
    pq_tmp.write_bytes(raw)
    pq_sha = sha256_file(pq_tmp)
    if pq_sha != pin["parquet_sha256"]:
        bad = out.with_suffix(".parquet.unpinned")
        pq_tmp.replace(bad)
        raise RuntimeError(f"STALE OR CHANGED DATA: {url} has sha256 {pq_sha}, but pins.json "
                           f"says {pin['parquet_sha256']}. {out} is unchanged; the download "
                           f"is in {bad}. Check the change before you update the pin.")
    rows = read_rows(pq_tmp)
    tmp = out.with_suffix(".jsonl.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            t = Task(
                id=r["id"], benchmark="lab-bench", subset=subset,
                subtask=r.get("subtask") or subset, question=r["question"],
                ideal=r["ideal"], distractors=tuple(r.get("distractors") or ()),
                canary=r.get("canary") or "",
            )
            f.write(json.dumps(t.to_dict(), ensure_ascii=False) + "\n")
    new_sha = sha256_file(tmp)
    if new_sha != pin["sha256"] or (pin.get("rows") and len(rows) != pin["rows"]):
        bad = out.with_suffix(".jsonl.unpinned")
        tmp.replace(bad)
        pq_tmp.unlink()
        raise RuntimeError(f"the parquet file matches its pin, but the JSONL made from it has "
                           f"sha256 {new_sha} and {len(rows)} rows (pinned {pin['sha256']}, "
                           f"{pin.get('rows')} rows). The conversion changed. {out} is "
                           f"unchanged; the new JSONL is in {bad}.")
    pq_tmp.replace(pq_path)
    tmp.replace(out)
    meta = {
        "source": f"hf://datasets/{pin['hf_dataset']}",
        "config": subset,
        "split": "train",
        "hf_revision": pin["hf_revision"],
        "parquet_url": url,
        "parquet_sha256": pq_sha,
        "license": "CC-BY-SA-4.0",
        "rows": len(rows),
        "sha256": new_sha,
        "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "note": "parquet file of the pinned commit; both SHA-256 values match pins.json.",
    }
    Path(str(out) + ".meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return out
