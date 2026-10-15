#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
BM25 稀疏召回探针：参考文献裁剪前 vs 裁剪后的对照实验

背景（这个探针是怎么来的）
--------------------------
给项目加混合检索前，先用真实语料核对稀疏侧的行为。结果**发现了问题**：
不裁参考文献时，查询「四联疗法 疗程」的 Top-5 全是 `[54]ChenQ, LongX…` 这类
引用条目，临床正文反而排到后面。原因是参考文献里期刊名/作者/药名高度密集，
BM25 的 idf 加权正好被它们刷满。

这个探针把「裁剪前 / 裁剪后」并排跑出来，用**引用污染率**量化差异：
    contamination@5 = Top-5 中「像引用条目」的块占比，越低越好

全程离线：PDF 解析、分块、分词、BM25 打分都不联网，不需要 API Key。
加密调用（embedding）才需要 Key —— 那部分是混合 vs 纯稠密的完整对比，另跑。

用法：
    cd RagHop
    python probe_sparse_recall.py
"""
from __future__ import annotations

import os
import re
import sys
import types

sys.stdout.reconfigure(encoding="utf-8")


def _import_app_with_config_stub():
    """主模块 import 时会构造 LLM 客户端，注入 config 桩即可复用其分块逻辑。"""
    stub = types.ModuleType("config")

    class Config:
        batch_size = 10
        model_name = "text-embedding-v3"
        dimensions = 1024
        api_key = "stub"
        base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        llm_api_key = "stub"
        llm_base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        llm_model = "qwen-plus"

    stub.Config = Config
    sys.modules["config"] = stub
    import homework3_langchain_app as app  # noqa: E402
    return app


# 判定「这块是参考文献条目」要按**条目结构**来，不能只看有没有出现引用编号：
# 正文里也常带行内引用（「耐药率为 60%~90%［8-11］」），那不算参考文献块。
# 参考文献条目的结构性标志是 [J]/[M] 文献类型标识、DOI、以及「年,卷(期):」著录格式。
_BIB_PAT = re.compile(
    r"［\s*[ＪＭＤＣ]\s*］|\[\s*[JMDCj]\s*\]"          # [J] 期刊 / [M] 专著 等文献类型标识
    r"|DOI\s*[:：]"                                    # DOI
    r"|\d{4}\s*[,，]\s*\d{1,3}\s*[(（]\s*\d{1,2}\s*[)）]\s*[:：]"  # 年,卷(期):
)


def is_citation_like(chunk: str) -> bool:
    """按著录结构判断是不是参考文献条目（结构性启发式，非语义判定）。"""
    return bool(_BIB_PAT.search(chunk))


QUERIES = [
    ("中文药名 · 四联疗法", "四联疗法 疗程 14 天"),
    ("中文专名 · 诊断指标", "甲胎蛋白 肝癌筛查"),
    ("英文缩写 · 蛋白名", "H.pylori 感染"),
    ("药名 · 新冠抗病毒", "奈玛特韦 利托那韦"),
    ("疾病专名", "多囊卵巢综合征 诊断标准"),
    ("纯语义问法（稀疏路不占优）", "感染了以后胃不舒服应该怎么治"),
]


def build_corpus(app, strip: bool):
    """返回 (chunks, sources)。strip=True 时先裁掉参考文献再分块。"""
    chunks, sources = [], []
    for name in sorted(os.listdir(app.DOC_DIR)):
        if not name.lower().endswith((".pdf", ".txt")):
            continue
        text = app.read_document(os.path.join(app.DOC_DIR, name))
        if strip:
            text = app.strip_references(text)
        cs = app.chunk_text(text, app.CHUNK_SIZE, app.CHUNK_OVERLAP)
        chunks.extend(cs)
        sources.extend([name] * len(cs))
    return chunks, sources


def main() -> int:
    app = _import_app_with_config_stub()
    from hybrid_retrieval import BM25, _HAS_JIEBA

    print("=" * 80)
    print("BM25 稀疏召回 · 参考文献裁剪对照实验（纯离线，无需 API Key）")
    print(f"分词后端: {'jieba' if _HAS_JIEBA else '字符 bigram'} | "
          f"分块: {app.CHUNK_SIZE}/{app.CHUNK_OVERLAP}")
    print("=" * 80)

    corpora = {}
    for strip in (False, True):
        label = "裁剪后" if strip else "未裁剪"
        chunks, sources = build_corpus(app, strip)
        corpora[strip] = (BM25(chunks), chunks, sources)
        print(f"  [{label}] 语料 {len(chunks):4d} 块，词表 {len(corpora[strip][0].idf):5d}")

    # ---------------- 逐条对照 ----------------
    for label, q in QUERIES:
        print()
        print("=" * 80)
        print(f"【{label}】 {q}")
        for strip in (False, True):
            bm25, chunks, sources = corpora[strip]
            tag = "裁剪后" if strip else "未裁剪"
            hits = bm25.top_n(q, 5)
            if not hits:
                print(f"  {tag}: （无命中）")
                continue
            n_cit = sum(is_citation_like(chunks[i]) for i, _ in hits)
            print(f"  {tag}（Top-5 中引用条目 {n_cit}/5）:")
            for rank, (idx, score) in enumerate(hits, 1):
                mark = "引" if is_citation_like(chunks[idx]) else "正"
                snippet = chunks[idx][:52].replace("\n", " ")
                print(f"     {rank}.[{mark}] {score:6.3f} {snippet}…")

    # ---------------- 汇总指标 ----------------
    print()
    print("=" * 80)
    print("汇总：引用污染率 contamination@5（Top-5 中引用条目的平均占比，越低越好）")
    print("=" * 80)
    for strip in (False, True):
        bm25, chunks, sources = corpora[strip]
        tag = "裁剪后" if strip else "未裁剪"
        rates = []
        for _, q in QUERIES:
            hits = bm25.top_n(q, 5)
            if not hits:
                continue
            rates.append(sum(is_citation_like(chunks[i]) for i, _ in hits) / len(hits))
        avg = sum(rates) / len(rates) if rates else 0.0
        print(f"  {tag}   {avg:.1%}   （{len(corpora[strip][1])} 块语料）")

    base = sum(
        sum(is_citation_like(corpora[False][1][i]) for i, _ in corpora[False][0].top_n(q, 5)) / 5
        for _, q in QUERIES) / len(QUERIES)
    after = sum(
        sum(is_citation_like(corpora[True][1][i]) for i, _ in corpora[True][0].top_n(q, 5)) / 5
        for _, q in QUERIES) / len(QUERIES)
    if base > 0:
        print()
        print(f"  → 引用污染率由 {base:.1%} 降到 {after:.1%}，"
              f"相对下降 {(base - after) / base:.0%}")

    print()
    print("结论")
    print("-" * 40)
    print("· 不裁参考文献时，稀疏路会被引用条目刷满高分 —— 这正是「照搬通用 RAG 分块")
    print("  直接上混合检索反而更差」的典型原因，必须先把分块污染清掉。")
    print("· 裁剪后 Top-5 落到临床正文，药名/专名的精确命中优势才真正体现出来。")
    print("· 纯语义问法（无专名）仍是稀疏路的短板 —— 那部分该由稠密路主导，")
    print("  两路互补正是 RRF 融合的意义。")
    print()
    print("注：本探针只刻画稀疏侧。混合 vs 纯稠密的端到端对比需要 embedding，")
    print("    配好 config.py 后用 HYBRID_ENABLED 开/关各跑一遍即可。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())