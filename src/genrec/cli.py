"""GenRec-ShortVideo 命令行入口。

用法（在项目根目录执行）：
  python -m genrec.cli download            # 下载并解压 KuaiRec
  python -m genrec.cli validate            # 校验原始数据（失败非零退出）
  python -m genrec.cli prepare             # 预处理（正式产物）
  python -m genrec.cli prepare --smoke     # 小规模 dry-run（产物带 _smoke 后缀）
"""

import argparse


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="genrec", description="GenRec-ShortVideo CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    for name, help_text in [
        ("download", "下载并解压 KuaiRec 数据（Zenodo 官方源）"),
        ("validate", "校验原始数据的规模/字段/覆盖率"),
    ]:
        p = sub.add_parser(name, help=help_text)
        p.add_argument("--config", default="configs/default.yaml")

    p_pre = sub.add_parser("prepare", help="预处理：训练样本/评估协议/内容特征")
    p_pre.add_argument("--config", default="configs/default.yaml")
    p_pre.add_argument("--smoke", action="store_true",
                       help="小规模 dry-run，产物带 _smoke 后缀")

    args = parser.parse_args(argv)

    from genrec.config import load_config
    cfg = load_config(args.config)

    if args.command == "download":
        from genrec.data import download
        download.run(cfg)
    elif args.command == "validate":
        from genrec.data import validate
        ok = validate.run(cfg)
        raise SystemExit(0 if ok else 1)
    elif args.command == "prepare":
        from genrec.data import preprocess
        preprocess.run(cfg, smoke=args.smoke)


if __name__ == "__main__":
    main()
