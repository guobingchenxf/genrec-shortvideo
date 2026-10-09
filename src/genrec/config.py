"""配置加载与路径解析。

配置文件位于 configs/ 目录下，项目根目录由其上级目录反推。
所有路径均为相对项目根的相对路径，通过 Config.path() 解析为绝对路径。
"""

from pathlib import Path

import yaml

DEFAULT_CONFIG = "configs/default.yaml"


class Config(dict):
    """轻量配置包装：dict 访问 + 项目根路径解析。"""

    def __init__(self, data, root):
        super().__init__(data)
        self.root = Path(root)

    def path(self, *keys):
        """按嵌套键取配置中的相对路径并解析为绝对路径。

        例：cfg.path("paths", "raw_dir") -> <root>/data/raw
        """
        node = self
        for key in keys:
            node = node[key]
        return (self.root / node).resolve()


def load_config(path=DEFAULT_CONFIG):
    path = Path(path)
    if path.parent.name == "configs":
        root = path.resolve().parent.parent
    else:
        root = Path.cwd()
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return Config(data, root)
