#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
混合检索（Hybrid Retrieval）：BM25 稀疏召回 + FAISS 稠密召回，用 RRF 融合

为什么要加这一层
----------------
原来的 `retrieve()` 只做稠密向量检索（FAISS 余弦 Top-5，见 homework3_langchain_app.py:152）。
稠密检索的问题在医疗这种场景尤其明显：**药名、指南编号、缩写这类精确词元**，
向量化之后语义相近但字面被稀释，很容易漏召。BM25 靠字面词频命中，正好补上这块。

融合用 RRF（Reciprocal Rank Fusion）而不是加权求和：
- RRF 只吃「排名」，不吃分数，所以不需要把余弦相似度和 BM25 分数归一化到同一量纲
  ——这两者的分布完全不可比，硬加权要调参且不稳定。
- 工业界常用做法，公式简单：score(d) = Σ 1/(k + rank_i(d))，k 默认 60。

分词
----
中文必须分词才能算 BM25。优先用 jieba；装不上时退化为**字符二元组**（bigram），
对中文短查询效果可接受，且零依赖——这样本模块在没装 jieba 的环境也能跑和单测。

本模块零第三方依赖（jieba 可选），可直接单独跑自测：
    python hybrid_retrieval.py
"""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Iterable, Sequence

# ---------------------------------------------------------------- 分词

try:
    import jieba  # type: ignore

    _HAS_JIEBA = True
except ImportError:
    _HAS_JIEBA = False

_ASCII_WORD = re.compile(r"[A-Za-z0-9_]+")
_CJK = re.compile(r"[一-鿿]")


def tokenize(text: str) -> list[str]:
    """中文分词 + 英文/数字按词切。返回小写词元列表。

    有 jieba 用 jieba；没有则中文按字符二元组切（bigram）。
    """
    if not text:
        return []

    text = text.lower()
    tokens: list[str] = []

    # 英文与数字：整词保留（药名缩写、指标名如 INR/CT 常是关键词）
    tokens.extend(_ASCII_WORD.findall(text))

    cjk_only = "".join(ch for ch in text if _CJK.match(ch))
    if not cjk_only:
        return tokens

    if _HAS_JIEBA:
        tokens.extend(w for w in jieba.lcut(cjk_only) if w.strip())
    else:
        # bigram：把「幽门螺杆菌」切成 幽门/门螺/螺杆/杆菌
        if len(cjk_only) == 1:
            tokens.append(cjk_only)
        else:
            tokens.extend(cjk_only[i:i + 2] for i in range(len(cjk_only) - 1))

    return tokens


# ---------------------------------------------------------------- BM25

class BM25:
    """Okapi BM25。纯 Python，索引规模在几百分块时完全够用。

    k1 控制词频饱和，b 控制文档长度归一化，取业界默认 1.5 / 0.75。
    """

    def __init__(self, corpus: Sequence[str], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus: list[str] = list(corpus)
        self.docs: list[list[str]] = [tokenize(d) for d in self.corpus]
        self.doc_len: list[int] = [len(d) for d in self.docs]
        self.avgdl: float = (sum(self.doc_len) / len(self.doc_len)) if self.doc_len else 0.0

        self.tf: list[Counter] = [Counter(d) for d in self.docs]

        df: Counter = Counter()
        for d in self.docs:
            df.update(set(d))
        n = len(self.docs)
        # BM25 的 idf 用带 +1 平滑的变体，避免高频词出现负值
        self.idf: dict[str, float] = {
            term: math.log(1 + (n - freq + 0.5) / (freq + 0.5))
            for term, freq in df.items()
        }

    def score(self, query: str, doc_idx: int) -> float:
        q_terms = tokenize(query)
        if not q_terms or not self.docs:
            return 0.0

        tf = self.tf[doc_idx]
        dl = self.doc_len[doc_idx] or 1
        total = 0.0
        for term in q_terms:
            f = tf.get(term)
            if not f:
                continue
            idf = self.idf.get(term, 0.0)
            denom = f + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1))
            total += idf * (f * (self.k1 + 1)) / denom
        return total

    def top_n(self, query: str, n: int) -> list[tuple[int, float]]:
        """返回 [(doc_idx, score), ...]，按分数降序，只保留正分。"""
        scored = [(i, self.score(query, i)) for i in range(len(self.docs))]
        scored = [(i, s) for i, s in scored if s > 0]
        scored.sort(key=lambda x: (-x[1], x[0]))
        return scored[:n]


# ---------------------------------------------------------------- RRF 融合

def rrf_fuse(rankings: Iterable[Sequence[int]], k: int = 60) -> list[tuple[int, float]]:
    """Reciprocal Rank Fusion。

    rankings: 多个「按相关性降序排列的 doc_idx 列表」
    返回 [(doc_idx, rrf_score), ...]，降序。

    score(d) = Σ_i 1 / (k + rank_i(d))，rank 从 1 开始。
    k=60 是 RRF 原论文的推荐值，起的是「压平头部」的作用：
    防止某一个检索器把它的第 1 名直接顶成最终第 1 名。
    """
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, doc_idx in enumerate(ranking, start=1):
            scores[doc_idx] = scores.get(doc_idx, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: (-x[1], x[0]))


# ---------------------------------------------------------------- 对外接口

class HybridRetriever:
    """把「稠密检索函数」和「BM25 语料」缝在一起。

    dense_search: 可调用对象，签名 (query, top_n) -> [(doc_idx, score), ...]
                  在 homework3 里就是把 FAISS 检索包一层，返回文档下标而不是原文。
    """

    def __init__(self, chunks: Sequence[str], dense_search, rrf_k: int = 60):
        self.chunks = list(chunks)
        self.dense_search = dense_search
        self.rrf_k = rrf_k
        self.bm25 = BM25(self.chunks)

    def search(self, query: str, top_k: int = 5, candidate_k: int = 20
               ) -> list[tuple[int, float, str]]:
        """混合召回。

        candidate_k 是每一路的候选池大小——先用较大的池子把候选捞出来，
        再用 RRF 收敛到 top_k。池子设成 top_k 的话混合就没有意义了。
        """
        candidate_k = max(candidate_k, top_k)

        dense_hits = self.dense_search(query, candidate_k)
        dense_ranking = [idx for idx, _ in dense_hits]

        sparse_ranking = [idx for idx, _ in self.bm25.top_n(query, candidate_k)]

        fused = rrf_fuse([dense_ranking, sparse_ranking], k=self.rrf_k)[:top_k]
        return [(idx, score, self.chunks[idx]) for idx, score in fused]


# ---------------------------------------------------------------- 自测

def _self_test() -> int:
    """零依赖自测：不联网、不需要 API Key、不需要 jieba。

    跑： python hybrid_retrieval.py
    """
    failures: list[str] = []

    def check(name: str, cond: bool):
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
        if not cond:
            failures.append(name)

    print(f"分词后端: {'jieba' if _HAS_JIEBA else '字符 bigram（未装 jieba）'}\n")

    print("1) 分词")
    toks = tokenize("幽门螺杆菌 HP 根除方案")
    check("非空", len(toks) > 0)
    check("保留英文词元 hp", "hp" in toks)

    print("\n2) BM25 排序：精确词元应当被顶上去")
    corpus = [
        "幽门螺杆菌感染的根除治疗方案推荐四联疗法，疗程十四天。",
        "原发性肝癌的筛查建议每六个月做一次甲胎蛋白与超声。",
        "多囊卵巢综合征的诊断标准与月经周期密切相关。",
        "新冠治疗的抗病毒药物使用时机与重症风险分层。",
        "糖尿病患者应定期监测血糖与糖化血红蛋白。",
    ]
    bm = BM25(corpus)
    hits = bm.top_n("幽门螺杆菌怎么根除", 3)
    check("召回非空", len(hits) > 0)
    check("第 1 名命中幽门螺杆菌文档", hits and hits[0][0] == 0)
    check("无关文档不入选", all(i != 4 for i, _ in hits))

    print("\n3) BM25 精确词元优势（这正是加稀疏检索的理由）")
    corpus2 = [
        "甲胎蛋白与超声联合用于肝癌筛查。",
        "肿瘤标志物的临床应用与解读。",
    ]
    bm2 = BM25(corpus2)
    r = bm2.top_n("甲胎蛋白", 2)
    check("缩写/专名精确命中", r and r[0][0] == 0 and r[0][1] > 0)

    print("\n4) RRF 融合")
    fused = rrf_fuse([[1, 2, 3], [2, 1, 3]], k=60)
    check("融合后覆盖全部文档", sorted(i for i, _ in fused) == [1, 2, 3])
    check("两路都排前的文档胜出", fused[0][0] in (1, 2))
    # 关键性质：同一排名下，「两路都命中」必须胜过「只被一路命中」。
    # 注意不能拿「A 的第 1 名」和「B 的第 1 名」比 —— 它们本就同分（都等于 1/(k+1)）。
    both_lists = dict(rrf_fuse([[1, 2, 3], [1, 2, 3]], k=60))
    one_list = dict(rrf_fuse([[1, 2, 3], [4, 5, 6]], k=60))
    check("双路命中 > 单路命中（同排名）", both_lists[1] > one_list[4])
    check("RRF 分数符合 1/(k+rank)", abs(both_lists[1] - 2.0 / 61) < 1e-12)

    print("\n5) HybridRetriever 端到端（用假稠密检索器）")
    def fake_dense(query, n):
        # 故意推一个 BM25 不会选的文档，验证融合确实在起作用
        return [(3, 0.9), (4, 0.8)][:n]

    hr = HybridRetriever(corpus, fake_dense, rrf_k=60)
    out = hr.search("幽门螺杆菌怎么根除", top_k=3, candidate_k=5)
    check("返回不超过 top_k", len(out) <= 3)
    idxs = [i for i, _, _ in out]
    check("稠密路候选被保留", 3 in idxs)
    check("稀疏路候选也被保留", 0 in idxs)
    check("返回的是原文", all(isinstance(t, str) and t for _, _, t in out))

    print()
    if failures:
        print(f"FAILED: {len(failures)} 项 -> {failures}")
        return 1
    print("全部通过。")
    return 0


if __name__ == "__main__":
    raise SystemExit(_self_test())