"""下载 KuaiRec 数据（多源策略；下载后由 validate 按官方口径校验后才用于实验）。

本机 2026-10-05 实测（写入 download_manifest.json）：
- 官方 Zenodo 整包直链速度约 42 KB/s（432MB 需数小时），不作为自动默认；
- HF 镜像（hf-mirror.com）上同一数据集的压缩矩阵速度约 455 KB/s；
- 官方 Zenodo 的小文件（caption / raw categories）可达且体积小。

默认路线：
  big_matrix.csv.gz, small_matrix.csv.gz          <- HF 镜像（非官方镜像；
      下载后按官方 README 统计口径严格校验行数/用户数/视频数/取值范围的
      一致性，校验通过才用于实验，报告写入 data/reports/）
  kuairec_caption_category.csv, video_raw_categories_multi.csv <- 官方 Zenodo

手动/官方完整包路线见 README；本模块不做任何静默降级：
尺寸不符 -> 抛错；校验口径不符 -> validate 阶段显式失败。
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
                raise IOError(
                    f"size mismatch: {part.stat().st_size} != {expected_bytes}")
            part.replace(dest)
            return dest
        except Exception as exc:  # noqa: BLE001 - 重试次数用尽后向上抛出
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
