# -*- coding: utf-8 -*-
"""
作业三（进阶路线）：基于 LangChain 的企业级多步骤 AI 问答应用
=====================================================================
对应课程：2-6 / 2-7 企业级教育智能问答助手｜掌握 LangChain 实现 AI 大模型应用

本文件综合运用「知识库 + 提示词 + LangChain 应用框架」，实现多步骤问答：
    用户提问 → 检索知识库 → 生成初步答案 → 检查并整理答案 → 返回最终结果

满足进阶路线 5 项要求：
    ① 使用 LangChain 调用大模型（ChatOpenAI，指向 dashscope 兼容接口）
    ② 设计两个提示词模板（初步答案生成 / 检查整理）
    ③ 完成两个连续处理步骤（LangChain LCEL 链路串联）
    ④ 调用知识库（本文件内嵌的 FAISS 检索，基于 医疗知识文档/ 下 PDF 构建）
    ⑤ 明确的输出格式 + 基础异常处理

运行方式：
    cd RagHop
    python homework3_langchain_app.py          # Web 界面（端口 7863，终端同步回显）
    python homework3_langchain_app.py --cli     # 纯终端：跑 3 类测试问题 + 问题记录
=====================================================================
"""

import os
import sys
import re
import json
import glob
import traceback

import faiss
import numpy as np
import fitz  # PyMuPDF，用于读取 PDF
from openai import OpenAI
from config import Config

# ===================== LangChain 相关依赖 =====================
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# 混合检索：BM25 稀疏召回 + 稠密召回，RRF 融合（零第三方依赖，可单独自测）
from hybrid_retrieval import HybridRetriever


# =====================================================================
# 0. 路径与基础参数
# =====================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOC_DIR = os.path.join(BASE_DIR, "医疗知识文档")      # 知识库文档目录（4 份临床指南 PDF）
INDEX_DIR = os.path.join(BASE_DIR, "indexes")         # 索引目录
os.makedirs(INDEX_DIR, exist_ok=True)

INDEX_PATH = os.path.join(INDEX_DIR, "homework3.index")       # 本作业 FAISS 向量索引
META_PATH = os.path.join(INDEX_DIR, "homework3_meta.json")    # 分块原文元数据

CHUNK_SIZE = 500        # 单个文档分块的最大字符数
CHUNK_OVERLAP = 50      # 相邻分块的重叠字符数
TOP_K = 5               # 检索时召回的片段数量

# 混合检索开关：稠密召回擅长语义近似，BM25 擅长药名/指南名这类精确词元。
# 关闭后回退成原来的纯稠密检索，方便对比两版效果。
HYBRID_ENABLED = True
CANDIDATE_K = 20        # 每一路的候选池大小（要比 TOP_K 大，否则融合没有意义）

# 建索引时裁掉文末参考文献。参考文献会污染检索（详见 strip_references 注释），
# 关掉可以复现「未裁剪」时的退化行为，用于对照。
STRIP_REFERENCES = True

# 复用 config.py 中的 API 配置：Embedding 与 LLM
embed_client = OpenAI(api_key=Config.api_key, base_url=Config.base_url)
llm_client = OpenAI(api_key=Config.llm_api_key, base_url=Config.llm_base_url)


# =====================================================================
# 1. 知识库检索（内嵌，FAISS + text-embedding-v3）
#    文档导入 → 分块 → Embedding → 向量存储 → 相似度检索
# =====================================================================
def read_document(path: str) -> str:
    if path.lower().endswith(".pdf"):
        doc = fitz.open(path)
        text = "".join(page.get_text() for page in doc)
    else:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
    return re.sub(r"\s+", " ", text).strip()


def chunk_text(text: str, chunk_size: int, chunk_overlap: int):
    sentences = re.split(r"(?<=[。！？；!?\n])", text)
    chunks, cur = [], ""
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        if len(s) > chunk_size:
            if cur:
                chunks.append(cur)
                cur = ""
            step = max(1, chunk_size - chunk_overlap)
            for i in range(0, len(s), step):
                piece = s[i:i + chunk_size]
                if piece:
                    chunks.append(piece)
            continue
        if len(cur) + len(s) <= chunk_size:
            cur += s
        else:
            if cur:
                chunks.append(cur)
            overlap = cur[-chunk_overlap:] if (chunk_overlap and cur) else ""
            cur = overlap + s
    if cur:
        chunks.append(cur)
    return [c.strip() for c in chunks if len(c.strip()) >= 10]


def embed(texts):
    if isinstance(texts, str):
        texts = [texts]
    vectors = []
    for i in range(0, len(texts), Config.batch_size):
        batch = texts[i:i + Config.batch_size]
        resp = embed_client.embeddings.create(
            model=Config.model_name,
            input=batch,
            dimensions=Config.dimensions,
            encoding_format="float",
        )
        vectors.extend([d.embedding for d in resp.data])
    arr = np.array(vectors, dtype=np.float32)
    faiss.normalize_L2(arr)
    return arr


def strip_references(text: str) -> str:
    """裁掉文末的参考文献列表。

    为什么必须裁：参考文献里期刊名、作者、药名高度密集，而正文里这些词出现次数有限。
    实测（probe_sparse_recall.py）不裁的话，BM25 的 Top-5 会被引用条目整段占满 ——
    查询「四联疗法 疗程」返回的全是 `[54]ChenQ, LongX…` 这种条目，临床正文反而排到后面。
    裁掉之后召回才落到真正的诊疗内容上。

    判定方式取**最后一次**出现「参考文献 / References」的位置，且必须出现在文档 30% 之后，
    避免正文里出现「检索了所有纳入文献研究的参考文献」这类字样时误伤。
    """
    matches = list(re.finditer(r"参考文献\s*[:：]?|^\s*References\s*$", text, re.M))
    if not matches:
        return text
    cut = matches[-1].start()
    if cut < len(text) * 0.3:
        return text
    return text[:cut]


def build_index(chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
    doc_paths = glob.glob(os.path.join(DOC_DIR, "*.pdf")) + \
                glob.glob(os.path.join(DOC_DIR, "*.txt"))
    if not doc_paths:
        raise FileNotFoundError(f"未在 {DOC_DIR} 找到任何 PDF/TXT 文档，请先放入知识库文档。")

    print(f"\n[步骤1-2] 导入并分块文档（chunk_size={chunk_size}, overlap={chunk_overlap}）")
    all_chunks = []
    for p in doc_paths:
        text = read_document(p)
        if STRIP_REFERENCES:
            trimmed = strip_references(text)
            if len(trimmed) < len(text):
                print(f"    - {os.path.basename(p)}：裁掉参考文献 "
                      f"{len(text) - len(trimmed)} 字（{100 * (1 - len(trimmed) / len(text)):.0f}%）")
            text = trimmed
        cs = chunk_text(text, chunk_size, chunk_overlap)
        print(f"    - {os.path.basename(p)}：{len(cs)} 个分块")
        all_chunks.extend(cs)

    print(f"[步骤3]  向量化 {len(all_chunks)} 个分块 ...")
    vectors = embed(all_chunks)

    print(f"[步骤4]  构建 FAISS 索引并保存 ...")
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    faiss.write_index(index, INDEX_PATH)
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)
    print(f"    索引构建完成，共 {len(all_chunks)} 个片段 -> {INDEX_PATH}\n")
    return len(all_chunks)


def retrieve(question: str, top_k: int = TOP_K):
    """检索知识库，返回 [(片段原文, 余弦相似度), ...]。

    默认走混合检索：稠密（FAISS 余弦）与稀疏（BM25）各取 CANDIDATE_K 个候选，
    用 RRF 融合排序后取前 top_k。**返回的分数依旧是真实余弦相似度** —— 按融合后的
    下标从 FAISS 里 reconstruct 出原向量重算，所以 CLI/Gradio 上打印的「相似度」
    与 cli_run_log.txt 里那组 0.827/0.801/0.784… 仍是同一个量纲，可直接对比。

    退回纯稠密检索：把 HYBRID_ENABLED 置 False。
    """
    index = faiss.read_index(INDEX_PATH)
    with open(META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)

    qv = embed(question)  # embed() 内部已做 L2 归一化

    def dense_search(q: str, n: int):
        """供混合检索当候选源：返回 [(doc_idx, 余弦), ...]。"""
        scores, ids = index.search(embed(q), n)
        return [(int(idx), float(scores[0][r]))
                for r, idx in enumerate(ids[0]) if 0 <= idx < len(meta)]

    if not HYBRID_ENABLED:
        return [(meta[idx], score) for idx, score in dense_search(question, top_k)]

    fused = HybridRetriever(meta, dense_search).search(
        question, top_k=top_k, candidate_k=CANDIDATE_K)

    hits = []
    for idx, _rrf_score, text in fused:
        # 用存进索引的原始向量重算余弦，保证对外的「相似度」口径不变
        cosine = float(np.dot(qv[0], index.reconstruct(idx)))
        hits.append((text, cosine))
    return hits


def ensure_index():
    if os.path.exists(INDEX_PATH) and os.path.exists(META_PATH):
        print("[初始化] 检测到已有知识库索引，直接加载。")
        return
    print("[初始化] 未找到索引，正在构建知识库（文档导入→分块→Embedding→FAISS）...")
    build_index()


# =====================================================================
# 2. LangChain 大模型（ChatOpenAI 指向阿里云百炼 / dashscope 兼容接口）
# =====================================================================
llm = ChatOpenAI(
    model=Config.llm_model,          # qwen-plus
    api_key=Config.llm_api_key,
    base_url=Config.llm_base_url,
    temperature=0.3,
    max_tokens=1024,
)


# =====================================================================
# 3. 两个提示词模板（满足进阶路线要求②）
# =====================================================================

# 模板 1：初步答案生成 —— 依据检索到的背景知识，生成第一版答案
DRAFT_SYSTEM = (
    "你是一名严谨的企业医疗知识问答助手，只依据用户提供的【背景知识】作答。"
    "若背景知识足以回答，请组织成通顺、准确的答案；"
    "若背景知识不足，请明确说明信息不足，不要编造。"
)
DRAFT_USER = (
    "【背景知识】\n{context}\n\n"
    "【用户问题】\n{question}\n\n"
    "请根据背景知识生成「初步答案」（仅基于资料，不编造）。"
)
draft_prompt = ChatPromptTemplate.from_messages([
    ("system", DRAFT_SYSTEM),
    ("user", DRAFT_USER),
])

# 模板 2：检查并整理 —— 质检初步答案、剔除幻觉、规范输出格式
POLISH_SYSTEM = (
    "你是答案质检与整理专家。请检查【初步答案】是否严格基于【背景知识】，"
    "若发现编造或背景中不存在的内容，必须删除或更正；"
    "然后将答案整理为下方规定的固定输出格式。"
)
POLISH_USER = (
    "【背景知识】\n{context}\n\n"
    "【用户问题】\n{question}\n\n"
    "【初步答案】\n{draft}\n\n"
    "请检查并整理，严格按以下格式输出（不要输出格式说明以外的多余内容）：\n"
    "【最终答案】\n（基于背景知识的规范回答；若信息不足则说明无法回答）\n\n"
    "【引用来源】\n（列出支撑答案的片段编号，如：片段1、片段3；若无则写“无”）\n\n"
    "【回答可信度】\n（高 / 中 / 低，并一句话说明依据）\n\n"
    "【补充说明】\n（给非专业用户的使用提示或免责声明；若无则写“无”）"
)
polish_prompt = ChatPromptTemplate.from_messages([
    ("system", POLISH_SYSTEM),
    ("user", POLISH_USER),
])

# 模板 3：自反思校验 —— 判断最终答案是否合格，决定是否回环重做（决策步骤）
#   产出结构化结论，便于程序解析驱动回环
JUDGE_SYSTEM = (
    "你是医疗问答答案的最终审核员。请判断【最终答案】是否满足下列要求："
    "（a）严格基于【背景知识】，无编造；"
    "（b）确实回答了【用户问题】；"
    "（c）若背景知识不足以回答，需明确标注为信息不足。"
    "请只输出结构化结论，便于程序解析。"
)
JUDGE_USER = (
    "【背景知识】\n{context}\n\n"
    "【用户问题】\n{question}\n\n"
    "【最终答案】\n{final}\n\n"
    "请按以下格式严格输出（不要多余内容）：\n"
    "【质检结论】（PASS / INSUFFICIENT / HALLUCINATED 三选一；"
    "PASS=合格；INSUFFICIENT=信息不足以回答；HALLUCINATED=存在编造）\n"
    "【原因】（一句话说明判断依据）\n"
    "【重写问题】（若结论非 PASS，给出一个更利于检索的重写问题；"
    "若 PASS 则写“无”）"
)
judge_prompt = ChatPromptTemplate.from_messages([
    ("system", JUDGE_SYSTEM),
    ("user", JUDGE_USER),
])


# =====================================================================
# 4. 三个连续处理步骤（满足进阶路线要求③：LangChain LCEL 链路）
#    步骤 A 生成 → 步骤 B 整理 → 步骤 C 校验（决策，驱动回环）
# =====================================================================
# 步骤 A：依据背景知识 → 生成初步答案
draft_chain = draft_prompt | llm | StrOutputParser()

# 步骤 B：结合「背景 + 初步答案」→ 检查整理 → 规范输出
polish_chain = polish_prompt | llm | StrOutputParser()

# 步骤 C：自反思校验 → 输出质检结论（PASS/INSUFFICIENT/HALLUCINATED）+ 重写问题
judge_chain = judge_prompt | llm | StrOutputParser()

# 回环上限：超过该轮数仍不通过则返回最后一次结果（避免无限循环）
MAX_ROUNDS = 3


def format_context(hits):
    if not hits:
        return "（未检索到相关背景知识）"
    return "\n\n".join(f"[片段{i+1}] {c}" for i, (c, _) in enumerate(hits))


def parse_judge(text: str):
    """解析校验链路的结构化输出，返回 (verdict, reason, rewrite)。"""
    verdict, reason, rewrite = "PASS", "", ""
    for line in (text or "").splitlines():
        s = line.strip()
        if s.startswith("【质检结论】"):
            v = s.replace("【质检结论】", "").strip().upper()
            if "INSUFFICIENT" in v:
                verdict = "INSUFFICIENT"
            elif "HALLUCINATED" in v or "HALLUC" in v:
                verdict = "HALLUCINATED"
            else:
                verdict = "PASS"
        elif s.startswith("【原因】"):
            reason = s.replace("【原因】", "").strip()
        elif s.startswith("【重写问题】"):
            rewrite = s.replace("【重写问题】", "").strip()
            if rewrite in ("无", "“无”", "「无」"):
                rewrite = ""
    return verdict, reason, rewrite


# =====================================================================
# 5. 主流程：完整多步骤（含自反思回环）
#    提问 → 检索 → 生成初步答案 → 检查整理 → 校验
#           ↑                                          │
#           └──── 若未通过：用重写问题回检索重做 ───────┘
#    通过 / 达到上限 → 返回最终结果
#    （含明确输出格式 + 基础异常处理，满足要求⑤）
# =====================================================================
def answer(question: str, top_k: int = TOP_K):
    question = (question or "").strip()
    if not question:
        return "⚠️ 请输入问题。", "", [], []

    ensure_index()

    cur_question = question
    rounds = []  # 每轮记录：{round, question, hits, draft, final, verdict, reason}
    final_out, draft_out, hits_out, trace_out = "", "", [], []

    for r in range(1, MAX_ROUNDS + 1):
        # ---- 步骤 1：检索知识库 ----
        try:
            hits = retrieve(cur_question, int(top_k))
        except Exception as e:
            return f"❌ 知识库检索失败：{e}", "", [], rounds
        context = format_context(hits)

        # ---- 步骤 2：生成初步答案（步骤 A）----
        try:
            draft = draft_chain.invoke({"context": context, "question": cur_question})
        except Exception as e:
            traceback.print_exc()
            return (
                f"❌ 大模型生成初步答案失败（可能是 API Key 失效或网络问题）：{e}",
                "", hits, rounds,
            )

        # ---- 步骤 3：检查并整理（步骤 B，连续第二步）----
        try:
            final = polish_chain.invoke({
                "context": context,
                "question": cur_question,
                "draft": draft,
            })
        except Exception as e:
            traceback.print_exc()
            return (
                f"【最终答案（整理步骤异常，降级返回初步答案）】\n{draft}",
                draft, hits, rounds,
            )

        # ---- 步骤 4：自反思校验（步骤 C，决策）----
        try:
            judge_text = judge_chain.invoke({
                "context": context,
                "question": cur_question,
                "final": final,
            })
            verdict, reason, rewrite = parse_judge(judge_text)
        except Exception as e:
            traceback.print_exc()
            # 校验失败不影响返回已整理答案
            verdict, reason, rewrite = "PASS", f"校验异常：{e}", ""

        rounds.append({
            "round": r,
            "question": cur_question,
            "verdict": verdict,
            "reason": reason,
            "draft": draft,
            "final": final,
        })
        final_out, draft_out, hits_out, trace_out = final, draft, hits, rounds

        # ---- 决策：通过则返回；否则用重写问题回检索重做 ----
        if verdict == "PASS":
            break
        if r < MAX_ROUNDS and rewrite:
            cur_question = rewrite  # 回环：用重写问题进入下一轮检索
            continue
        # 无重写问题或已达上限，直接返回最后一次结果
        break

    return final_out, draft_out, hits_out, trace_out


# =====================================================================
# 6. 测试问题（满足任务 C：覆盖三类场景）
# =====================================================================
TEST_QUESTIONS = [
    "幽门螺杆菌感染的推荐治疗方案是什么？",                 # ① 正常且信息完整
    "幽门螺杆菌根除失败后的处理与原发性肝癌的筛查手段分别是什么？",  # ② 需结合多个知识片段
    "高血压的一线降压药物有哪些？",                         # ③ 知识库中无答案
]


def run_cli():
    ensure_index()
    for i, q in enumerate(TEST_QUESTIONS, 1):
        print("\n" + "=" * 72)
        print(f"测试问题 {i}：{q}")
        print("-" * 72)
        final, draft, hits, rounds = answer(q)
        print("【多步骤回环轨迹】")
        for rd in rounds:
            print(f"  第{rd['round']}轮：问题=\"{rd['question'][:40]}...\" → "
                  f"校验={rd['verdict']}（{rd['reason'][:40]}）")
        print(f"\n【召回片段（相似度，共{len(hits)}条）】")
        for j, (c, s) in enumerate(hits, 1):
            print(f"  {j}. (相似度 {s:.3f}) {c[:70].replace(chr(10),' ')}...")
        print("\n【初步答案】\n" + draft)
        print("\n【最终答案（检查整理后）】\n" + final)
    print("\n" + "=" * 72)
    print("任务 C 问题记录（详见 README）：")
    print("问题：初步答案阶段模型易照搬背景、偶发格式不稳定，且单步无法自我纠正。")
    print("修改：①检查整理模板；②新增自反思校验模板 + 回环重做；③固定输出格式 + 异常降级。")
    print("结果：多步骤带决策回环，输出统一为四段格式，幻觉与格式问题被抑制。")


# =====================================================================
# 7. Web 界面（Gradio，端口 7863；终端同步回显）
# =====================================================================
def main_web():
    import gradio as gr

    def answer_fn(question: str, top_k: int):
        if not question or not question.strip():
            return "⚠️ 请输入问题", "", "", ""
        final, draft, hits, rounds = answer(question, int(top_k))

        hits_md = "### 📄 召回的文档片段\n\n" + (
            "\n\n".join(
                f"**片段 {i+1}**（相似度 `{s:.3f}`）\n\n{c}"
                for i, (c, s) in enumerate(hits, 1)
            ) if hits else "（无召回片段）"
        )

        trace_md = "### 🔄 多步骤回环轨迹\n\n" + "\n".join(
            f"**第{rd['round']}轮**　问题：`{rd['question']}`\n"
            f"- 校验结论：`{rd['verdict']}`　原因：{rd['reason']}"
            for rd in rounds
        ) if rounds else "（无轨迹）"

        print("\n" + "=" * 72)
        print(f"[终端] 问题：{question}（Top K={int(top_k)}）")
        print("【多步骤回环轨迹】")
        for rd in rounds:
            print(f"  第{rd['round']}轮：校验={rd['verdict']}（{rd['reason'][:50]}）")
        print("【召回片段】")
        for i, (c, s) in enumerate(hits, 1):
            print(f"  {i}. (相似度 {s:.3f}) {c[:70].replace(chr(10),' ')}...")
        print("\n【初步答案】\n" + draft)
        print("\n【最终答案】\n" + final)
        print("=" * 72)
        return hits_md, draft, final, trace_md

    with gr.Blocks(title="RagHop：基于 LangChain 的多步骤医疗问答") as demo:
        gr.Markdown(
            "## RagHop：基于 LangChain 的多步骤医疗问答\n"
            "流程：**提问 → 检索 → 生成初步答案 → 检查整理 → 自反思校验 →（不通过则重写问题回检索）→ 最终结果**\n\n"
            "使用 LangChain 调用 qwen-plus，复用本地医疗知识库（FAISS）；含自反思回环，最多 "
            f"{MAX_ROUNDS} 轮。"
        )
        with gr.Row():
            with gr.Column(scale=2):
                q_input = gr.Textbox(
                    label="问题",
                    lines=2,
                    placeholder="例如：幽门螺杆菌感染的推荐治疗方案是什么？",
                )
                top_k = gr.Slider(1, 10, value=TOP_K, step=1, label="召回片段数 Top K")
                submit_btn = gr.Button("提交", variant="primary")
                gr.Examples(TEST_QUESTIONS, q_input)
            with gr.Column(scale=3):
                trace_out = gr.Markdown(label="多步骤回环轨迹", value="（提交后显示）")
                hits_out = gr.Markdown(label="召回的文档片段", value="（提交后显示）")
                draft_out = gr.Markdown(label="初步答案（步骤一）", value="（提交后显示）")
                final_out = gr.Markdown(label="最终答案（检查整理后）", value="（提交后显示）")
        submit_btn.click(
            answer_fn,
            inputs=[q_input, top_k],
            outputs=[trace_out, hits_out, draft_out, final_out],
        )

    ensure_index()
    print("[启动] Gradio 界面地址： http://0.0.0.0:7863")
    demo.launch(server_name="0.0.0.0", server_port=7863, share=False)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--cli":
        run_cli()
    else:
        main_web()
