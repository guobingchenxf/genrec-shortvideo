"""GenRec-ShortVideo 命令行入口。

用法（在项目根目录执行）：
  python -m genrec.cli download            # 下载并解压 KuaiRec
  python -m genrec.cli validate            # 校验原始数据（失败非零退出）
  python -m genrec.cli prepare [--smoke]   # 预处理（产物/干跑分离）
  python -m genrec.cli train-rqvae [--smoke]          # 训练语义 ID（RQ-VAE）
  python -m genrec.cli train-gen --variant sid|raw|raw-gru [--smoke]
  python -m genrec.cli evaluate [--methods pop,itemcf,...] [--smoke]
  python -m genrec.cli generate --user-id 0 --topk 10
"""

import argparse

# 与 genrec.train.VARIANTS 及 configs 的 gen.variants 保持一致（有测试防漂移）
TRAIN_GEN_VARIANTS = ["sid", "sid-v2", "sid-b1", "sid-b3", "raw", "raw-gru"]


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="genrec", description="GenRec-ShortVideo CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    for name, help_text in [
        ("download", "下载并解压 KuaiRec 数据"),
        ("validate", "校验原始数据的规模/字段/覆盖率"),
        ("prepare", "预处理：训练样本/评估协议/内容特征"),
    ]:
        p = sub.add_parser(name, help=help_text)
        p.add_argument("--config", default="configs/default.yaml")
    sub.choices["prepare"].add_argument(
        "--smoke", action="store_true", help="小规模 dry-run，产物带 _smoke 后缀")

    p_rq = sub.add_parser("train-rqvae", help="训练语义 ID（RQ-VAE）")
    p_rq.add_argument("--config", default="configs/default.yaml")
    p_rq.add_argument("--smoke", action="store_true")
    p_rq.add_argument("--tag", default=None,
                      help="实验标签（如 v2，读取 models.rqvae_v2 配置并区分产物）")

    p_ce = sub.add_parser("encode-content",
                          help="B1：神经内容编码器特征（bge-small-zh）")
    p_ce.add_argument("--config", default="configs/default.yaml")
    p_ce.add_argument("--out", default="content_feats_bge.npz")

    p_tg = sub.add_parser("train-gen", help="训练生成式序列模型")
    p_tg.add_argument("--variant", choices=TRAIN_GEN_VARIANTS,
                      required=True)
    p_tg.add_argument("--config", default="configs/default.yaml")
    p_tg.add_argument("--smoke", action="store_true")

    p_ev = sub.add_parser("evaluate", help="全观测协议评估")
    p_ev.add_argument("--methods",
                      default="pop,itemcf,gen-sid,gen-sid-v2,gen-raw,gru-raw")
    p_ev.add_argument("--config", default="configs/default.yaml")
    p_ev.add_argument("--smoke", action="store_true")
    p_ev.add_argument("--beam", type=int, default=None,
                      help="覆盖生成方法的 beam 大小（消融用）")

    p_sp = sub.add_parser("evaluate-sampled",
                          help="标准协议评估（留一法 + 100 负采样）")
    p_sp.add_argument("--methods",
                      default="random,pop,itemcf,gen-sid,gen-sid-v2,"
                              "gen-raw,gru-raw")
    p_sp.add_argument("--max-users", type=int, default=None,
                      help="只评前 N 个用户（调试/加速用；默认全量）")
    p_sp.add_argument("--n-neg", type=int, default=100)
    p_sp.add_argument("--seed", type=int, default=42)
    p_sp.add_argument("--config", default="configs/default.yaml")

    p_gn = sub.add_parser("generate", help="单用户生成 demo")
    p_gn.add_argument("--user-id", type=int, required=True)
    p_gn.add_argument("--topk", type=int, default=10)
    p_gn.add_argument("--beam", type=int, default=10)
    p_gn.add_argument("--config", default="configs/default.yaml")

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
    elif args.command == "train-rqvae":
        from genrec import train
        train.train_rqvae_main(cfg, smoke=args.smoke, tag=args.tag)
    elif args.command == "encode-content":
        from genrec.data import content_neural
        content_neural.run(cfg, out_name=args.out)
    elif args.command == "train-gen":
        from genrec import train
        train.run_gen(cfg, variant=args.variant, smoke=args.smoke)
    elif args.command == "evaluate":
        from genrec.eval import run_eval
        run_eval.run(cfg, methods=args.methods.split(","), smoke=args.smoke,
                     beam=args.beam)
    elif args.command == "evaluate-sampled":
        from genrec.eval import sampled
        sampled.run(cfg, methods=args.methods.split(","),
                    max_users=args.max_users, n_neg=args.n_neg, seed=args.seed)
    elif args.command == "generate":
        from genrec import generate
        generate.run(cfg, user_id=args.user_id, topk=args.topk,
                     beam=args.beam)


if __name__ == "__main__":
    main()
