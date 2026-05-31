import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import ollama

# =====================
# LOAD VECTOR DB
# =====================
model = SentenceTransformer("keepitreal/vietnamese-sbert")
index = faiss.read_index("medical.index")

with open("medical_metadata.pkl", "rb") as f:
    data = pickle.load(f)

questions = data["questions"]
diseases = data["diseases"]


def search(query, k=5):

    query_vector = model.encode([query]).astype("float32")
    distances, indices = index.search(query_vector, k)

    context = []

    for i in indices[0]:
        context.append(
            f"Bệnh: {diseases[i]} - Triệu chứng: {questions[i]}"
        )

    return context


def ask_llm(query, context):

    context_text = "\n".join(context)

    prompt = f"""
Bạn là bác sĩ AI.

Dựa vào dữ liệu sau:
{context_text}

Câu hỏi người dùng:
{query}

Hãy:
- đoán bệnh có khả năng nhất
- giải thích
- hướng điều trị
- trả lời tự nhiên tiếng Việt
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


def chatbot():

    print("Medical AI Local Chatbot")
    print("------------------------")

    while True:

        query = input("\nBạn: ")

        if query == "exit":
            break

        context = search(query)

        answer = ask_llm(query, context)

        print("\nAI:", answer)


chatbot()