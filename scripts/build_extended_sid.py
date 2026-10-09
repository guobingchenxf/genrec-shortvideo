import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from genrec.config import load_config
from genrec.utils.monitor import save_json


def build_extended(codes):
    """codes: (N, L) -> (N, L+1)，追加组内序号使 SID 全局唯一。"""
    groups = defaultdict(list)
    for i, key in enumerate(map(tuple, codes.tolist())):
        groups[key].append(i)
    ext = np.zeros(len(codes), dtype=np.int16)
    for idxs in groups.values():
        for rank, i in enumerate(sorted(idxs)):
            ext[i] = rank
    return np.concatenate([codes, ext[:, None]], axis=1)


def main():
    cfg = load_config()
    processed = cfg.path("paths", "processed_dir")
    z = np.load(processed / "sid_codes_v2.npz")
    video_ids, codes = z["video_ids"], z["codes"]
    ext_codes = build_extended(codes)

    # 验证：扩展后唯一性
    uniq = len({tuple(c) for c in ext_codes.tolist()})
    assert uniq == len(ext_codes), "扩展后仍存在碰撞，逻辑有误"

    np.savez_compressed(processed / "sid_codes_b3.npz",
                        video_ids=video_ids, codes=ext_codes)
    collisions_before = len(ext_codes) - len(
        {tuple(c) for c in codes.tolist()})
    stats = {
        "source": "sid_codes_v2.npz",
        "levels_before": int(codes.shape[1]),
        "levels_after": int(ext_codes.shape[1]),
        "videos": len(ext_codes),
        "unique_sids_after": int(uniq),
        "collided_videos_before": int(collisions_before),
        "ext_code_max": int(ext_codes[:, -1].max()),
        "note": "组内按 video_id 升序追加序号；扩展后解码无需展开候选组",
    }
    save_json(stats, cfg.root / "results" / "experiments"
              / "sid_extended_stats.json")
    print(f"[b3] {stats}")
    print(f"[b3] -> {processed / 'sid_codes_b3.npz'}")


if __name__ == "__main__":
    main()
