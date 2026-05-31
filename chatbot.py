import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import ollama
from predict_skin import predict_image

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

        print("\nChọn chế độ:")
        print("1. Hỏi bằng văn bản")
        print("2. Chẩn đoán qua ảnh da")
        print("exit. Thoát")

        mode = input("\nBạn chọn: ")

        if mode == "exit":
            break

        # =========================
        # CHAT TEXT
        # =========================
        if mode == "1":

            query = input("\nBạn: ")

            context = search(query)

            answer = ask_llm(
                query,
                context
            )

            print("\nAI:", answer)

        # =========================
        # IMAGE DETECTION
        # =========================
        elif mode == "2":

            path = input(
                "\nNhập đường dẫn ảnh: "
            )

            try:

                results = predict_image(path)

                print("\nTop 3 bệnh gần nhất:")

                for disease, score in results:

                    print(
                        f"- {disease}: {round(score * 100, 2)}%"
                    )

                # lấy bệnh cao nhất
                top_disease = results[0][0]

                query = (
                    f"Tôi có dấu hiệu bệnh da "
                    f"{top_disease}"
                )

                context = search(query)

                answer = ask_llm(
                    query,
                    context
                )

                print("\nAI:", answer)

            except Exception as e:

                print(
                    "\nLỗi đọc ảnh:",
                    e
                )

        else:

            print(
                "\nVui lòng chọn 1 / 2 / exit"
            )


chatbot()