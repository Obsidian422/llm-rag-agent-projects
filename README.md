# llm-rag-agent-projects

> 检索增强生成（RAG）的工程实践。以 LangChain 链路编排为基础，在一个真实语料
> （医疗临床指南）上做**可量化的检索质量改进**：混合检索（BM25 + 稠密 + RRF）、
> 参考文献污染治理，并用带标注的评测集把「到底改进了多少」量出来。

## 目录

| 目录 | 内容 |
|---|---|
| [`RagHop/`](RagHop/) | 基于 LangChain 的多步骤医疗问答应用。LCEL 三步链路（生成 → 整理 → 自反思校验）+ 决策回环，本地 FAISS 知识库，混合检索 |

## 起点与增量

这个仓库**起步于一门 LangChain 课程的练习题**——`RagHop/homework3_langchain_app.py`
头部 docstring 里保留了当时的作业说明。课程要求的是一条多步骤问答链路。

之后的增量是自己加的，也是这个仓库主要想展示的部分：

1. **混合检索**（`RagHop/hybrid_retrieval.py`）
   BM25 稀疏召回 + FAISS 稠密召回，用 RRF 融合。选 RRF 而不是加权求和，是因为
   余弦相似度与 BM25 分数的分布不可比，硬加权需要归一化调参且不稳定；RRF 只吃
   排名，不需要统一量纲。中文分词优先 jieba，无 jieba 时退化为字符二元组。

2. **参考文献污染治理**（`strip_references()`）
   用真实语料核对稀疏召回时暴露出的问题。不裁参考文献时，查询「四联疗法 疗程」
   的 Top-5 全是 `[54]ChenQ, LongX…` 这类引用条目，临床正文被挤到后面——参考文献里
   期刊名/作者/药名高度密集，正好把 BM25 的 idf 加权刷满。实测参考文献占幽门螺杆菌
   指南 **52%**、肝癌指南 **64%** 的文本；裁剪后索引由 **573 块降至 270 块**。

3. **对照评测**（`RagHop/eval_hybrid_vs_dense.py`）
   同一索引、同一批问题，只切 `HYBRID_ENABLED` 开关做 A/B：

   | ground truth 层级 | 规模 | 纯稠密 | 混合检索 |
   |---|---|---|---|
   | 文档级（该问题属于哪份指南） | 16 题标注 | Hit@5 100% | Hit@5 100% |
   | 片段级（稀有精确词元所在片段） | 100 个词元，词频 1–5 全量抽样，固定种子 42 | **Hit@5 16.0% / MRR 0.0995** | **Hit@5 84.0% / MRR 0.6983** |

   文档级两法都满分——任务太简单，区分不出差异；把 ground truth 下沉到片段级
   才量出稀疏路的真实价值：它集中在「只出现在极少数片段里的精确词元」，也就是
   向量检索会被语义近邻淹没的地方。**这个「先量、再下结论」的顺序是刻意的**，
   否则「我觉得加了更好」正是简历上最容易被追问穿帮的一类说法。

4. **离线探针**（`RagHop/probe_sparse_recall.py`）
   PDF 解析、分块、分词、BM25 打分全程不联网、不需要 API Key，用来独立刻画
   稀疏侧行为，并做参考文献裁剪前后的对照（指标：`contamination@5`）。

## 快速开始

```bash
cd RagHop
pip install -r requirements.txt
cp config.example.py config.py        # 填入自己的阿里云百炼 API Key
python homework3_langchain_app.py     # Web 界面，端口 7863
```

无需 API Key 的可复现入口（离线）：

```bash
cd RagHop
python probe_sparse_recall.py         # 稀疏侧行为 + 参考文献裁剪对照
python hybrid_retrieval.py            # 混合检索模块自测
```

需要 API Key 的完整对照评测：

```bash
cd RagHop
python eval_hybrid_vs_dense.py        # 混合检索 vs 纯稠密，片段级 Hit@K / MRR
```

## 关于开发方式

本仓库的代码与文档在 **Claude Code 辅助下**完成，提交信息里保留了
`Co-Authored-By: Claude Code` 的署名。

分工大致是：链路选型、检索引擎取舍（为什么是 RRF 不是加权）、评测集的设计
（为什么文档级不够、要下沉到片段级），以及参考文献污染这类问题的发现与定位，
是自己做的判断；AI 承担的是实现、重构与文档整理的吞吐。评测脚本、探针与
`cli_run_log.txt` 都是可以自己重跑的。

## License

仅供学习与研究使用。知识库文档（`RagHop/医疗知识文档/`）版权归各自发布机构所有。