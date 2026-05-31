import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import ollama

# =====================
# CONFIG
# =====================
TOP_K_SEARCH = 10
TOP_K_SHOW = 3
MIN_SCORE = 0.75   # 75%

# =====================
# LOAD VECTOR DB
# =====================
model = SentenceTransformer("keepitreal/vietnamese-sbert")

index = faiss.read_index("medical.index")

with open("medical_metadata.pkl", "rb") as f:
    data = pickle.load(f)

questions = data["questions"]
diseases = data["diseases"]


# =====================
# convert distance -> %
# =====================
def similarity_from_distance(distance):
    # vì đang dùng embedding normalized
    score = 1 / (1 + float(distance))
    return round(score * 100, 2)


# =====================
# SEARCH
# =====================
def search(query):

    query_vector = model.encode(
        [query],
        normalize_embeddings=True
    ).astype("float32")

    distances, indices = index.search(
        query_vector,
        TOP_K_SEARCH
    )

    results = []

    for dist, idx in zip(distances[0], indices[0]):

        if idx == -1:
            continue

        score_percent = similarity_from_distance(dist)

        if score_percent >= MIN_SCORE * 100:
            results.append({
                "disease": diseases[idx],
                "symptom": questions[idx],
                "score": score_percent
            })

    # sort giảm dần
    results = sorted(
        results,
        key=lambda x: x["score"],
        reverse=True
    )

    return results[:TOP_K_SHOW]


# =====================
# LLM
# =====================
def ask_llm(query, results):

    if not results:
        return (
            "Xin lỗi, tôi chưa tìm thấy bệnh nào "
            "đủ độ chính xác để gợi ý từ dữ liệu hiện tại."
        )

    context_text = "\n".join([
        f"- {r['disease']} ({r['score']}%): {r['symptom']}"
        for r in results
    ])

    prompt = f"""
Bạn là chatbot tư vấn y tế.

Dữ liệu tìm được:
{context_text}

Người dùng hỏi:
{query}

Yêu cầu:
- chỉ dựa trên dữ liệu trên
- ưu tiên bệnh có % cao nhất
- giải thích ngắn gọn
- liệt kê tối đa 3 bệnh
- trả lời tiếng Việt tự nhiên
- nếu chưa đủ chắc chắn thì nói cần khám bác sĩ
"""

    response = ollama.chat(
        model="llama3.2",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response["message"]["content"]


# =====================
# CHATBOT
# =====================
def chatbot():

    print("Medical AI Local Chatbot")
    print("------------------------")

    while True:

        query = input("\nBạn: ")

        if query == "exit":
            break

        results = search(query)

        # debug
        if results:
            print("\nTop kết quả:")
            for r in results:
                print(
                    f"{r['disease']} - {r['score']}%"
                )
        else:
            print("\nKhông có kết quả phù hợp.")

        answer = ask_llm(query, results)

        print("\nAI:", answer)


chatbot()