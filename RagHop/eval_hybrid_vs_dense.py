#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
混合检索 vs 纯稠密检索 · 对照评测

评什么
------
文档级检索准确率：给定问题，Top-K 召回的片段是否来自**该问题正确的那份指南**。
    Hit@1 / Hit@5 / MRR
用文档级而不是片段级做 ground truth，是因为「哪一段最相关」有主观性，
而「这个问题该由哪份指南回答」是客观的、可复核的。

为什么要有这个脚本
------------------
hybrid_retrieval.py 的自测只证明 BM25 与 RRF 的实现正确。
probe_sparse_recall.py 只刻画稀疏侧、且证明了参考文献污染的存在。
**「加了混合检索到底有没有变好」必须用带标注的数据集在完整链路上量出来**，
否则就是把「我觉得更好」当成结论 —— 那正是简历上最容易被追问穿帮的一类说法。

对照组
------
同一份索引、同一批问题，只切 HYBRID_ENABLED：
    纯稠密 = False   （改动前的原始行为）
    混合   = True    （FAISS 余弦 + BM25，RRF 融合）

用法：
    cd RagHop
    python eval_hybrid_vs_dense.py
"""
from __future__ import annotations

import os
import sys
import types

sys.stdout.reconfigure(encoding="utf-8")

# ---------------------------------------------------------------------------
# 标注集：问题 -> 期望命中的指南文件
# 混合了「专名/药名」（稀疏路占优）与「语义问法」（稠密路占优）两类，
# 以便看清两路各自在什么情况下起作用。
# ---------------------------------------------------------------------------
HP = "中国幽门螺杆菌感染治疗指.pdf"
LC = "原发性肝癌诊疗指南.pdf"
PCOS = "多囊卵巢综合征中国诊疗指南.pdf"
COV = "新冠治疗方案.pdf"

GOLD = [
    # —— 含精确专名/药名，预期稀疏路有优势 ——
    ("H.pylori 感染初次根除推荐什么方案？", HP),
    ("铋剂四联方案的疗程是几天？", HP),
    ("阿莫西林耐药时怎么调整用药？", HP),
    ("甲胎蛋白用于肝癌的哪个环节？", LC),
    ("HBV 抗病毒治疗能降低肝癌风险吗？", LC),
    ("TACE 治疗适用于哪些肝癌患者？", LC),
    ("PCOS 的诊断标准是什么？", PCOS),
    ("多囊卵巢综合征患者高雄激素怎么处理？", PCOS),
    ("奈玛特韦利托那韦的用法用量？", COV),
    ("新冠重型患者的治疗原则是什么？", COV),

    # —— 纯语义问法，不含专名，预期稠密路主导 ——
    ("胃里查出细菌感染，该怎么治？", HP),
    ("哪些人需要定期做肝脏方面的检查？", LC),
    ("月经不规律又长痘，可能是什么问题？", PCOS),
    ("发烧咳嗽去医院一般怎么处理？", COV),
    ("抗生素耐药了还有别的办法吗？", HP),
    ("这个病可以用介入手术治疗吗？", LC),
]


def _import_app_with_config_stub():
    stub = types.ModuleType("config")
    import config as real
    stub.Config = real.Config
    sys.modules["config"] = stub
    import homework3_langchain_app as app  # noqa: E402
    return app


def build_text2doc(app) -> tuple[dict, int]:
    """chunk 原文 -> 来源文件名。

    索引的 meta 只存了片段文本、没存来源，所以用「同一套分块逻辑重跑一遍」建映射。
    按文本匹配而不是按下标，这样不受 glob 顺序影响。
    """
    import glob
    paths = glob.glob(os.path.join(app.DOC_DIR, "*.pdf")) + glob.glob(os.path.join(app.DOC_DIR, "*.txt"))
    mapping, dup = {}, 0
    for p in paths:
        text = app.read_document(p)
        if app.STRIP_REFERENCES:
            text = app.strip_references(text)
        for c in app.chunk_text(text, app.CHUNK_SIZE, app.CHUNK_OVERLAP):
            if c in mapping:
                dup += 1
            else:
                mapping[c] = os.path.basename(p)
    return mapping, dup


def rank_of_expected(docs: list[str], expected: str) -> int | None:
    for i, d in enumerate(docs, 1):
        if d == expected:
            return i
    return None


def evaluate(app, text2doc: dict, hybrid: bool, k: int = 5, candidate_k: int = 20) -> dict:
    app.HYBRID_ENABLED = hybrid

    rows = []
    for q, expected in GOLD:
        hits = app.retrieve(q, k)                       # 走真实检索入口
        docs = [text2doc.get(text, "?") for text, _ in hits]
        r = rank_of_expected(docs, expected)
        rows.append({"q": q, "expected": expected, "docs": docs,
                     "rank": r, "top1": docs[0] if docs else "?"})

    n = len(rows)
    return {
        "hit@1": sum(1 for r in rows if r["rank"] == 1) / n,
        "hit@5": sum(1 for r in rows if r["rank"] is not None) / n,
        "mrr": sum((1.0 / r["rank"]) if r["rank"] else 0.0 for r in rows) / n,
        "rows": rows,
    }


def rare_term_eval(app, hybrid: bool, n_terms: int = 100, seed: int = 42,
                   df_max: int = 5, k: int = 5) -> dict:
    """片段级评测：稀有精确词元的召回。

    为什么用这个设计
    ----------------
    文档级评测（上面那个）只有 4 份文档，稠密检索随手就能命中正确文档，两种方法都打满，
    区分不出差异。而稀疏检索真正的用武之地是**只出现在极少数片段里的精确词元**
    （药名、指标名、英文缩写）—— 这类词在向量空间里会被大量语义相近的片段淹没。

    为了不落入「挑词挑到混合检索赢为止」，这里**机械地定义样本**：
    取词表里文档频率 1~df_max 的全部词元，固定随机种子抽样，不做任何人工筛选。
    问法就是这个词元本身 —— 问题退化为「这个词出现在哪一块」，
    即纯粹考察字面定位能力。
    """
    import json
    import random
    from collections import Counter

    from hybrid_retrieval import tokenize

    meta = json.load(open(app.META_PATH, encoding="utf-8"))
    df = Counter()
    for c in meta:
        df.update(set(tokenize(c)))

    rare = sorted(t for t, d in df.items() if 1 <= d <= df_max and len(t) >= 2)
    random.Random(seed).shuffle(rare)
    terms = rare[:n_terms]

    app.HYBRID_ENABLED = hybrid

    ranks = []
    for t in terms:
        gold = {meta[i] for i, c in enumerate(meta) if t in c}
        hits = app.retrieve(t, k)
        rank = None
        for pos, (text, _) in enumerate(hits, 1):
            if text in gold:
                rank = pos
                break
        ranks.append(rank)

    n = len(ranks)
    return {
        "n_terms": n,
        "vocab_rare_size": len(rare),
        "hit@1": sum(1 for r in ranks if r == 1) / n if n else 0.0,
        "hit@5": sum(1 for r in ranks if r) / n if n else 0.0,
        "mrr": sum((1.0 / r) if r else 0.0 for r in ranks) / n if n else 0.0,
        "n_miss": sum(1 for r in ranks if not r),
        "sample_terms": terms[:15],
    }


def main() -> int:
    app = _import_app_with_config_stub()
    app.ensure_index()

    text2doc, dup = build_text2doc(app)
    print(f"片段->来源映射: {len(text2doc)} 条（重复文本 {dup} 条）")
    print(f"索引片段数: {len(app.retrieve('测试', 1))} 可见 | HYBRID_ENABLED={app.HYBRID_ENABLED}, "
          f"CANDIDATE_K={app.CANDIDATE_K}, TOP_K={app.TOP_K}")
    print()

    results = {}
    for label, hybrid in (("纯稠密", False), ("混合检索", True)):
        print(f"跑 {label} …")
        results[label] = evaluate(app, text2doc, hybrid)

    # ---------------- 汇总 ----------------
    print()
    print("=" * 78)
    print(f"文档级检索准确率（n={len(GOLD)} 问，Top-K={app.TOP_K}）")
    print("=" * 78)
    print(f"{'':12s}{'Hit@1':>10s}{'Hit@5':>10s}{'MRR':>10s}")
    for label in ("纯稠密", "混合检索"):
        r = results[label]
        print(f"{label:12s}{r['hit@1']:>10.1%}{r['hit@5']:>10.1%}{r['mrr']:>10.4f}")

    d, h = results["纯稠密"], results["混合检索"]
    print()
    print(f"Hit@1  Δ = {(h['hit@1'] - d['hit@1']) * 100:+.1f} 个百分点")
    print(f"Hit@5  Δ = {(h['hit@5'] - d['hit@5']) * 100:+.1f} 个百分点")
    print(f"MRR    Δ = {h['mrr'] - d['mrr']:+.4f}")

    # ---------------- 逐题对比（只看两版结论不同的） ----------------
    print()
    print("=" * 78)
    print("逐题差异（两版命中排名不同的题）")
    print("=" * 78)
    changed = 0
    for rd, rh in zip(d["rows"], h["rows"]):
        if rd["rank"] != rh["rank"]:
            changed += 1
            fmt = lambda r: f"rank={r}" if r else "未命中"
            print(f"· {rd['q']}")
            print(f"    期望 {rd['expected'][:22]}")
            print(f"    纯稠密 {fmt(rd['rank'])}   top1={rd['top1'][:22]}")
            print(f"    混合   {fmt(rh['rank'])}   top1={rh['top1'][:22]}")
    if not changed:
        print("（两版逐题排名完全一致）")

    # ---------------- 实验二：片段级 · 稀有精确词元 ----------------
    print()
    print("=" * 78)
    print("实验二 · 片段级：稀有精确词元的召回（词频 1–5，机械抽样 100 个）")
    print("=" * 78)
    rare_res = {}
    for label, hybrid in (("纯稠密", False), ("混合检索", True)):
        print(f"跑 {label} …")
        rare_res[label] = rare_term_eval(app, hybrid)
    rd, rh = rare_res["纯稠密"], rare_res["混合检索"]
    print(f"样本词元 {rd['n_terms']} 个（稀有词表共 {rd['vocab_rare_size']} 个）")
    print(f"抽样示例: {rd['sample_terms'][:8]}")
    print()
    print(f"{'':12s}{'Hit@1':>10s}{'Hit@5':>10s}{'MRR':>10s}{'未命中':>8s}")
    for label in ("纯稠密", "混合检索"):
        r = rare_res[label]
        print(f"{label:12s}{r['hit@1']:>10.1%}{r['hit@5']:>10.1%}{r['mrr']:>10.4f}{r['n_miss']:>8d}")
    print()
    print(f"Hit@1  Δ = {(rh['hit@1'] - rd['hit@1']) * 100:+.1f} 个百分点")
    print(f"Hit@5  Δ = {(rh['hit@5'] - rd['hit@5']) * 100:+.1f} 个百分点")
    print(f"MRR    Δ = {rh['mrr'] - rd['mrr']:+.4f}")

    print()
    print("解读注意")
    print("-" * 40)
    print("· 实验一（文档级）标注客观但太粗：仅 4 份文档，稠密检索容易打满，")
    print("  两法可能都到 100%，区分不出差异 —— 这种情况下**不能**说混合检索更好。")
    print("· 实验二的样本是机械定义的（词频 1–5 全量抽样），不做人工挑选，")
    print("  避免「挑到混合检索赢为止」；但问法退化成单个词元，属压力测试而非真实场景。")
    print("· 结论要两个实验一起看；任何单边有利的数字都不足以支撑简历上的改进声明。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())