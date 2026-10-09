"""防漂移测试：CLI 选项、train.VARIANTS、config 的 gen.variants 三方必须一致。

"""

import pytest

from genrec.cli import TRAIN_GEN_VARIANTS
from genrec.config import load_config
from genrec.train import VARIANTS


def test_three_sources_of_variants_agree():
    cfg = load_config()  # 项目根存在 configs/default.yaml
    config_variants = set(cfg["models"]["gen"]["variants"].keys())
    assert set(TRAIN_GEN_VARIANTS) == set(VARIANTS.keys()) == config_variants, (
        f"variant 来源不一致：cli={sorted(TRAIN_GEN_VARIANTS)}, "
        f"train={sorted(VARIANTS)}, config={sorted(config_variants)}")


def test_sid_variants_have_sid_files_mapping():
    from genrec.train import SID_FILES
    sid_variants = {v for v, is_sid in VARIANTS.items() if is_sid}
    assert sid_variants <= set(SID_FILES.keys()), (
        f"SID 变体缺少语义 ID 文件映射：{sid_variants - set(SID_FILES)}")


def test_eval_methods_cover_all_generator_variants():
    """评估侧方法表必须覆盖全部生成模型变体（历史上三次接入遗漏）。"""
    from genrec.eval.run_eval import METHOD_VARIANT
    from genrec.eval.sampled import DEFAULT_METHODS

    assert set(METHOD_VARIANT.values()) <= set(VARIANTS.keys()), (
        f"评估方法引用了未定义的变体："
        f"{set(METHOD_VARIANT.values()) - set(VARIANTS)}")
    covered = set(METHOD_VARIANT)
    for method in covered:
        assert method in DEFAULT_METHODS, f"{method} 未进入采样协议默认方法表"


def test_sasrec_enters_both_protocols():
    """C2 防漂移：SASRec 基线必须同时出现在两种协议的默认方法表中。"""
    from genrec.eval.run_eval import ALL_METHODS
    from genrec.eval.sampled import DEFAULT_METHODS

    assert "sasrec" in ALL_METHODS
    assert "sasrec" in DEFAULT_METHODS


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
