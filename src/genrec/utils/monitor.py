"""资源与耗时实测工具。

诚实性原则：报告里出现的耗时与内存数字必须来自这里量到的值。
峰值内存在步骤边界采样（未安装 psutil 时不记录，不估算）。
"""

import json
import time
from pathlib import Path

try:
    import psutil
except ImportError:  # psutil 仅用于资源实测记录，缺失时静默降级
    psutil = None


class StepTimer:
    """记录每个步骤的墙钟耗时与进程峰值内存。"""

    def __init__(self):
        self.steps = {}
        self.peak_rss_mb = None
        self._t0 = None
        self._name = None
        self._proc = psutil.Process() if psutil is not None else None

    def _sample_peak(self):
        if self._proc is not None:
            rss_mb = self._proc.memory_info().rss / 1e6
            if self.peak_rss_mb is None or rss_mb > self.peak_rss_mb:
                self.peak_rss_mb = rss_mb

    def start(self, name):
        if self._name is not None:
            self.stop()
        self._name = name
        self._t0 = time.perf_counter()

    def stop(self):
        if self._name is None:
            return
        self.steps[self._name] = round(time.perf_counter() - self._t0, 2)
        self._sample_peak()
        self._name = None

    def report(self):
        if self._name is not None:
            self.stop()
        out = {"wall_seconds": dict(self.steps)}
        if self.peak_rss_mb is not None:
            out["peak_rss_mb_at_step_boundaries"] = round(self.peak_rss_mb, 1)
        return out


def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
