"""
Bước 2 — Prompt Hub & A/B Routing (COMPLETE SOLUTION)
===================================
"""
from qa_pairs import SAMPLE_QUESTIONS
from utils.data_loader import load_knowledge_base, split_text, build_vectorstore
from utils.llm_factory import get_llm, get_embeddings
from langsmith import Client, traceable
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import config  # ⚠️ phải import trước LangChain
import sys
import hashlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


# ── 1. Tên Prompt trên Hub ─────────────────────────────────────────────────
PROMPT_V1_NAME = "my-rag-prompt-v1"
PROMPT_V2_NAME = "my-rag-prompt-v2"


# ── 2. Định nghĩa 2 Prompt Templates ──────────────────────────────────────
# V1: CONCISE (ngắn gọn, 2-4 câu)
SYSTEM_V1 = """Bạn là trợ lý AI hữu ích. Dùng CHỈ thông tin từ context dưới để trả lời câu hỏi. 
Giữ câu trả lời ngắn gọn (2-4 câu), chính xác, không thêm thông tin ngoài context. 
Nếu context không đủ để trả lời, hãy nói rõ ràng."""

PROMPT_V1 = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_V1),
    ("human",  "Context:\n{context}\n\nCâu hỏi: {question}"),
])

# V2: STRUCTURED (có cấu trúc, expert tone, 3-5 câu)
SYSTEM_V2 = """Bạn là chuyên gia AI với kinh nghiệm sâu. Đọc kỹ context và xác định tất cả facts liên quan.
Viết câu trả lời rõ ràng, có tổ chức, và chi tiết (3-5 câu). Nêu rõ cơ sở dựa trên context.
Nếu cần, tóm tắt các điểm chính. Không bổ sung thông tin ngoài context."""

PROMPT_V2 = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_V2),
    ("human",  "Context:\n{context}\n\nCâu hỏi: {question}"),
])


# ── 3. Push Prompts lên Prompt Hub ─────────────────────────────────────────
def push_prompts_to_hub(client: Client):
    """
    Upload cả 2 prompt templates lên LangSmith Prompt Hub.
    """
    # Push V1
    try:
        url = client.push_prompt(
            PROMPT_V1_NAME,
            object=PROMPT_V1,
            description="V1 – Ngắn gọn (2-4 câu) - Concise and direct responses"
        )
        print(f"✅ Đã push V1 → {url}")
    except Exception as e:
        print(f"⚠️  V1 lỗi: {e}")

    # Push V2
    try:
        url = client.push_prompt(
            PROMPT_V2_NAME,
            object=PROMPT_V2,
            description="V2 – Có cấu trúc (3-5 câu) - Structured expert-tone responses"
        )
        print(f"✅ Đã push V2 → {url}")
    except Exception as e:
        print(f"⚠️  V2 lỗi: {e}")


# ── 4. Pull Prompts từ Prompt Hub ──────────────────────────────────────────
def pull_prompts_from_hub(client: Client) -> dict:
    """
    Tải 2 prompt từ LangSmith Prompt Hub.
    Fallback về template local nếu Hub không khả dụng.
    """
    prompts = {}

    # Pull V1
    try:
        prompts[PROMPT_V1_NAME] = client.pull_prompt(PROMPT_V1_NAME)
        print(f"↓ Đã pull '{PROMPT_V1_NAME}' từ Hub")
    except Exception as e:
        print(f"ℹ️  Dùng local fallback cho '{PROMPT_V1_NAME}': {e}")
        prompts[PROMPT_V1_NAME] = PROMPT_V1

    # Pull V2
    try:
        prompts[PROMPT_V2_NAME] = client.pull_prompt(PROMPT_V2_NAME)
        print(f"↓ Đã pull '{PROMPT_V2_NAME}' từ Hub")
    except Exception as e:
        print(f"ℹ️  Dùng local fallback cho '{PROMPT_V2_NAME}': {e}")
        prompts[PROMPT_V2_NAME] = PROMPT_V2

    return prompts


# ── 5. A/B Routing tất định ────────────────────────────────────────────────
def get_prompt_version(request_id: str) -> str:
    """
    Xác định prompt version dựa trên MD5 hash của request_id.

    Quy tắc: hash chẵn → PROMPT_V1_NAME | hash lẻ → PROMPT_V2_NAME
    TÍNH CHẤT: cùng request_id LUÔN cho cùng kết quả (deterministic).
    """
    # Tính MD5 hash của request_id và chuyển thành số nguyên
    hash_int = int(hashlib.md5(request_id.encode()).hexdigest(), 16)

    # Trả về V1 nếu chẵn, V2 nếu lẻ
    return PROMPT_V1_NAME if hash_int % 2 == 0 else PROMPT_V2_NAME


# ── 6. Traced A/B Query ────────────────────────────────────────────────────
@traceable(name="ab-rag-query", tags=["ab-test", "step2"])
def ask_ab(retriever, llm, prompt, question: str, version: str) -> dict:
    """
    Chạy RAG chain với prompt version được chọn bởi router.

    Bước:
      a) Retrieve top-3 docs từ retriever
      b) Ghép page_content thành context string
      c) Chạy chain
      d) Trả về dict kết quả
    """
    # Retrieve docs từ retriever
    docs = retriever.invoke(question)

    # Ghép page_content thành 1 string
    context = "\n\n".join([doc.page_content for doc in docs])

    # Chạy chain
    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({
        "context": context,
        "question": question
    })

    # Trả về dict kết quả
    return {
        "question": question,
        "answer": answer,
        "version": version,
        "context_length": len(context),
        "num_docs": len(docs)
    }


# ── 7. Setup Vectorstore (tái sử dụng logic Bước 1) ───────────────────────
def setup_vectorstore():
    embeddings = get_embeddings()
    text = load_knowledge_base()
    chunks = split_text(text)
    return build_vectorstore(chunks, embeddings)


# ── 8. Main ────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Bước 2: Prompt Hub & A/B Routing")
    print("=" * 60)

    if not config.validate():
        sys.exit(1)

    # Tạo LangSmith Client
    client = Client(api_key=config.LANGSMITH_API_KEY)

    # Push cả 2 prompts lên Hub
    push_prompts_to_hub(client)

    # Pull cả 2 prompts từ Hub (dùng dict trả về)
    prompts = pull_prompts_from_hub(client)

    # Tạo vectorstore, retriever và LLM
    vectorstore = setup_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm = get_llm()

    # Chạy A/B routing cho tất cả câu hỏi
    v1_count, v2_count = 0, 0
    print("\n" + "=" * 60)
    print("Running A/B Test on 50 Questions")
    print("=" * 60)

    for i, question in enumerate(SAMPLE_QUESTIONS):
        request_id = f"req-{i:04d}"

        # Lấy version key từ request_id qua get_prompt_version()
        version_key = get_prompt_version(request_id)
        version_tag = "v1" if version_key == PROMPT_V1_NAME else "v2"
        prompt = prompts[version_key]

        # Gọi ask_ab() với đúng arguments
        result = ask_ab(retriever, llm, prompt, question, version_tag)

        if version_tag == "v1":
            v1_count += 1
        else:
            v2_count += 1

        print(f"[{i+1:02d}] [prompt-{version_tag}] {question[:55]}...")

    print("\n" + "=" * 60)
    print(f"📊 Routing Statistics:")
    print(f"  V1 (Concise):   {v1_count} questions")
    print(f"  V2 (Structured): {v2_count} questions")
    print(f"  Total:          {len(SAMPLE_QUESTIONS)} questions")
    print("=" * 60)
    print("✅ Bước 2 hoàn thành!")
    print("📍 Kiểm tra trên LangSmith:")
    print(f"  • Prompts: https://smith.langchain.com/hub")
    print(f"  • Traces:  https://smith.langchain.com/projects/")


if __name__ == "__main__":
    main()
