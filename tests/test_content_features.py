"""合成小文件测试 build_content_features（不依赖真实数据）。

合成数据仅用于测试代码正确性。
"""

import pandas as pd

from genrec.data.preprocess import build_content_features


def test_build_content_features_synthetic(tmp_path):
    cap = pd.DataFrame({
        "video_id": ["1", "2", "3"],
        "manual_cover_text": ["", "cover", ""],
        "caption": ["狗狗视频 #tag", "", ""],
        "topic_tag": ["[狗]", "", ""],
        "first_level_category_name": ["宠物", "宠物", ""],
        "second_level_category_name": ["狗", "猫", ""],
        "third_level_category_name": ["", "", ""],
    })
    cap.to_csv(tmp_path / "kuairec_caption_category.csv", index=False)
    cats = pd.DataFrame({
        "video_id": ["1", "3"],
        "category_name": ["宠物", "美食"],
    })
    cats.to_csv(tmp_path / "video_raw_categories_multi.csv", index=False)

    cfg = {"tfidf_ngram_range": [2, 3], "tfidf_max_features": 1000,
           "tfidf_min_df": 1, "svd_dim": 4}
    ids, feats, coverage, _evr = build_content_features(
        [1, 2, 3, 4], tmp_path, cfg, seed=0)

    assert ids.tolist() == [1, 2, 3, 4]
    assert feats.shape == (4, 4)
    assert coverage["total_videos"] == 4
    # with_caption = caption 文件里有非空文本（含封面/话题/类目名）
    assert coverage["with_caption"] == 2     # 视频 1、2
    assert coverage["with_categories"] == 2  # 视频 1、3
    assert coverage["empty"] == 1            # 视频 4 无任何内容


def test_build_content_features_handles_missing_cells(tmp_path):
    # 字段缺失（NaN）不应导致类型错误（真实数据中 caption 文件有 8 行缺失）
    cap = pd.DataFrame({
        "video_id": ["1"],
        "caption": ["x"],
        "topic_tag": [None],
        "first_level_category_name": [None],
        "second_level_category_name": [None],
        "third_level_category_name": [None],
        "manual_cover_text": [None],
    })
    cap.to_csv(tmp_path / "kuairec_caption_category.csv", index=False)
    pd.DataFrame({"video_id": ["1"], "category_name": ["美食"]}).to_csv(
        tmp_path / "video_raw_categories_multi.csv", index=False)

    cfg = {"tfidf_ngram_range": [2, 3], "tfidf_max_features": 100,
           "tfidf_min_df": 1, "svd_dim": 1}
    ids, feats, _coverage, _evr = build_content_features(
        [1], tmp_path, cfg, seed=0)
    assert ids.tolist() == [1]
    assert feats.shape == (1, 1)
