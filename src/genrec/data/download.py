"""下载 KuaiRec 数据
"""

import gzip
import hashlib
import shutil
import time
import urllib.request
from pathlib import Path

from genrec.utils.monitor import save_json


def sha256_file(path, chunk_size=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk_size)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def download_with_resume(url, dest, expected_bytes=None, retries=3):
    """带断点续传的下载：先写 dest.part，完成后原子改名为 dest。"""
    dest = Path(dest)
    part = Path(str(dest) + ".part")
    for attempt in range(1, retries + 1):
        offset = part.stat().st_size if part.exists() else 0
        headers = {"Range": f"bytes={offset}-"} if offset else {}
        req = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                resume_ok = offset > 0 and resp.status == 206
                mode = "ab" if resume_ok else "wb"
                with open(part, mode) as f:
                    while True:
                        block = resp.read(1 << 20)
                        if not block:
                            break
                        f.write(block)
            if expected_bytes is not None and part.stat().st_size != expected_bytes:
                raise OSError(
                    f"size mismatch: {part.stat().st_size} != {expected_bytes}")
            part.replace(dest)
            return dest
        except Exception as exc:
            print(f"[download] attempt {attempt}/{retries} failed: {exc}")
            if attempt == retries:
                raise
            time.sleep(3)
    return dest


def _decompress_gz(src, dst):
    with gzip.open(src, "rb") as fin, open(dst, "wb") as fout:
        shutil.copyfileobj(fin, fout, length=1 << 22)


def run(cfg):
    raw_dir = cfg.path("paths", "raw_dir")
    data_dir = raw_dir / cfg["data"]["dir_name"]
    data_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = cfg.path("paths", "reports_dir")

    manifest = {
        "dataset": "KuaiRec (Kuaishou short-video recommendation, CIKM 2022)",
        "route_note": (
            "Default route: big/small matrices from hf-mirror.com (unofficial "
            "mirror of the same dataset; validated against official statistics "
            "by `validate`), caption/raw-categories from official Zenodo. "
            "Official zip measured ~42 KB/s from this network on 2026-10-05; "
            "manual official route documented in README."),
        "files": {},
    }
    t_all = time.perf_counter()
    for fname, spec in cfg["data"]["files"].items():
        target = data_dir / fname
        is_gz = bool(spec.get("gz", False))
        local = data_dir / (fname + ".gz") if is_gz else target
        expected = spec.get("compressed_bytes")

        t0 = time.perf_counter()
        cached = (local.exists() and expected is not None
                  and local.stat().st_size == expected)
        if not cached:
            print(f"[download] fetching {fname} <- {spec['url']}")
            download_with_resume(spec["url"], local, expected_bytes=expected)
        else:
            print(f"[download] cached: {local.name}")

        if is_gz and not target.exists():
            print(f"[download] decompressing {local.name} -> {target.name}")
            _decompress_gz(local, target)

        manifest["files"][fname] = {
            "url": spec["url"],
            "source": spec["source"],
            "was_cached": cached,
            "compressed_bytes_on_disk": local.stat().st_size,
            "sha256_of_local_file": sha256_file(local),
            "decompressed_bytes": (target.stat().st_size
                                   if target.exists() else None),
            "wall_seconds": round(time.perf_counter() - t0, 2),
        }
    manifest["total_wall_seconds"] = round(time.perf_counter() - t_all, 2)

    out = reports_dir / "download_manifest.json"
    save_json(manifest, out)
    for fname, info in manifest["files"].items():
        print(f"[download] {fname}: {info['decompressed_bytes']} bytes "
              f"(sha256 {info['sha256_of_local_file'][:16]}...)")
    print(f"[download] manifest -> {out}")
    return data_dir
