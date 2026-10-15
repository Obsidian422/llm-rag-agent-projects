# RagHop

> 基于 LangChain 的多步骤医疗问答应用。场景是企业员工的健康/用药咨询，知识库为
> `医疗知识文档/` 下 4 份临床诊疗指南 PDF，要求答案可溯源、信息不足时拒答而不是编造。

**起点说明**：本目录起步于一门 LangChain 课程的练习题，入口脚本沿用当时的文件名
`homework3_langchain_app.py`，头部 docstring 也保留了作业说明。课程要求的是一条
「多步骤链路」；**混合检索、参考文献污染治理、对照评测与离线探针都是之后自己加的增量**，
详见根目录 [README](../README.md)。

---

## 1. 架构

```
用户提问
   ↓
retrieve()  ── 混合检索：FAISS 稠密 Top-N  ∥  BM25 稀疏 Top-N
   │                    └── RRF 融合 → Top-K 片段
   ↓
draft_chain   生成初步答案（只依据【背景知识】，不编造）
   ↓
polish_chain  检查并整理（剔除幻觉、规范格式、标注引用与可信度）
   ↓
judge_chain   自反思校验 → PASS / INSUFFICIENT / HALLUCINATED + 重写问题
   ↓
   ├─ PASS 或已达 MAX_ROUNDS=3 → 返回最终结果
   └─ 未通过 → 用重写问题回到 retrieve() 重做（决策回环）
```

| 层 | 选型 |
|---|---|
| 编排 | LangChain LCEL 声明式链路（`prompt \| llm \| StrOutputParser()`） |
| 大模型 | `qwen-plus`，经 `ChatOpenAI` 指向阿里云百炼 dashscope 兼容接口 |
| Embedding | `text-embedding-v3`（1024 维） |
| 向量库 | FAISS `IndexFlatIP`（余弦相似度） |
| 稀疏检索 | 自实现 BM25（`hybrid_retrieval.py`，零第三方依赖） |
| 融合 | RRF，`k=60` |
| 界面 | Gradio，端口 7863（`share=False`） |

## 2. 关键设计决策

**为什么三步 + 回环，而不是一步生成。**
单提示词直接生成时，模型会把背景里的无关细节一并输出、信息不足时硬凑（幻觉）、
输出格式不稳定。加第二级「检查并整理」解决了格式与幻觉，但线性两步**没有决策能力**
——发现信息不足也无法重试。第三级 `judge_chain` 输出结构化结论
（`PASS` / `INSUFFICIENT` / `HALLUCINATED` + 重写问题），驱动主流程回环重做，
这才是多步骤与普通串联的分水岭。`MAX_ROUNDS=3` 防止无限循环。

**为什么 RRF 融合而不是加权求和。**
余弦相似度和 BM25 分数的分布完全不可比，硬加权要先归一化、要调参、且不稳定。
RRF 只吃**排名**不吃分数，`score(d) = Σ 1/(k + rank_i(d))`，无需统一量纲，是工业界
常规做法。`k=60` 取原论文推荐值，起「压平头部」的作用。

**召回池必须大于 top_k。**
融合前两路各自的候选池取 `max(top_k, …)`；池子如果设成 `top_k`，融合就退化成
「在同一个集合上重排序」，混合检索失去意义。

**相似度仍然报真实余弦值。**
融合后按下标从 FAISS `reconstruct` 出原向量重算余弦，所以 CLI/Gradio 上打印的
「相似度」与 `cli_run_log.txt` 里的那组数值（0.827 / 0.801 / …）仍是同一量纲，可直接对比。

**中文分词的分级降级。**
优先 jieba；装不上时退化为字符二元组（bigram），对中文短查询效果可接受且零依赖，
保证模块在无 jieba 的环境也能跑自测。

## 3. 参考文献污染

用真实语料核对稀疏侧行为时暴露出的问题：不裁参考文献时，查询「四联疗法 疗程」的
Top-5 **全是** `[54]ChenQ, LongX…` 这类引用条目，临床正文反被挤到后面。原因是参考文献里
期刊名/作者/药名高度密集，正好把 BM25 的 idf 加权刷满。

- 实测参考文献占幽门螺杆菌指南 **52%**、肝癌指南 **64%** 的文本
- `strip_references()` 取最后一次出现「参考文献 / References」的位置，且要求位于文档
  **30% 之后**，避免正文里的同名字样被误伤
- 裁剪后索引由 **573 块降至 270 块**（直接减少了近一半的检索噪声）

## 4. 评测

`eval_hybrid_vs_dense.py`：同一份索引、同一批问题，只切 `HYBRID_ENABLED` 做 A/B。

| ground truth 层级 | 规模 | 纯稠密 | 混合检索 |
|---|---|---|---|
| 文档级（该问题属于哪份指南） | 16 题人工标注 | Hit@5 100% | Hit@5 100% |
| 片段级（稀有精确词元所在片段） | 100 个词元，词频 1–5 全量抽样，`seed=42` | **Hit@5 16.0% / MRR 0.0995** | **Hit@5 84.0% / MRR 0.6983** |

两点值得说明：

- **文档级两法都满分，说明这个层面的任务区分不出差异。** 把 ground truth 下沉到片段级，
  才量出稀疏路的价值集中在「只出现在极少数片段里的精确词元」——也就是向量检索会被
  语义近邻淹没的地方。用文档级数据下结论会得出「混合检索没用」的错误判断。
- **稀有词元是「全量抽样 + 固定种子」，不是人工挑的。** 人工挑选容易不自觉地挑出
  对自己方案有利的样本，固定种子抽样是可复核的前提。

## 5. 可复现入口

```bash
cd RagHop
pip install -r requirements.txt
cp config.example.py config.py     # 填入自己的阿里云百炼 API Key
```

| 命令 | 是否需 API Key | 产出 |
|---|---|---|
| `python homework3_langchain_app.py` | 需要 | Gradio Web 界面（端口 7863），三栏并排展示【召回片段 / 初步答案 / 最终答案】 |
| `python homework3_langchain_app.py --cli` | 需要 | 纯终端跑 3 类测试问题 + 打印多步骤回环轨迹 |
| `python eval_hybrid_vs_dense.py` | 需要（embedding） | 混合检索 vs 纯稠密的 Hit@K / MRR 对照 |
| `python hybrid_retrieval.py` | 不需要 | 模块自测（BM25 / RRF / 融合断言） |
| `python probe_sparse_recall.py` | 不需要 | 稀疏侧行为 + 参考文献裁剪前后对照（`contamination@5`） |

首次运行会自动构建索引到 `indexes/`（`homework3.index` + `homework3_meta.json`），
已存在则跳过。

## 6. 目录

```
RagHop/
├── homework3_langchain_app.py   # 入口：LCEL 三步链路 + 决策回环 + Gradio
├── hybrid_retrieval.py          # BM25 + RRF 融合；含自测，零第三方依赖
├── eval_hybrid_vs_dense.py      # 混合 vs 纯稠密 对照评测
├── probe_sparse_recall.py       # 稀疏侧离线探针 + 参考文献裁剪对照
├── config.example.py            # 配置模板（复制为 config.py）
├── cli_run_log.txt              # --cli 模式的真实运行记录
├── requirements.txt
├── 医疗知识文档/                # 4 份临床指南 PDF（知识库源文档）
└── indexes/                     # FAISS 索引，首次运行自动生成（未入库）
```

> `config.example.py` 里 `bert_path` / `recall_way` / `topd` 等字段是课程脚手架遗留的，
> 当前链路不使用，实际生效的是 `api_key` / `llm_api_key` / `base_url` / `model_name` /
> `dimensions` 这几项。

## 7. 已知局限

- `temperature=0.3`，回环轮数与重写问题文本每次运行会略有差异；机制与最终结论稳定，
  但复现具体轮次不保证逐字一致。
- 知识库只有 4 份指南，覆盖面窄；`probe` 与评测的语料规模也相应有限。
- 参考文献裁剪用的是「最后一次出现关键词且位于 30% 之后」的启发式，对排版异常的
  PDF 仍可能误伤；更稳的做法是解析 PDF 的章节结构或用分块器直接丢弃明显是引用格式的块。

## License

仅供学习与研究使用。`医疗知识文档/` 下文档版权归各自发布机构所有。