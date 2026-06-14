import base64
import faiss
import pickle
import random
import numpy as np
from sentence_transformers import SentenceTransformer
import ollama
from pathlib import Path
import tempfile

# =====================
# LOAD VECTOR DB
# =====================
BASE_DIR = Path(__file__).resolve().parent

SAMPLE_PROMPTS = {
    "general": [
        "Tôi hiện đang có các triệu chứng như sốt cao, đau đầu và mệt mỏi. Tôi có thể đang bị bệnh gì?",
        "Tôi đang cảm thấy ho kéo dài, đau họng và sổ mũi. Tôi có thể đang bị bệnh gì?",
        "Tôi hiện đang có các triệu chứng như đau bụng, tiêu chảy và buồn nôn. Tôi có thể đang bị bệnh gì?",
        "Tôi hay bị chóng mặt, hoa mắt khi đứng dậy đột ngột. Tôi có thể đang bị bệnh gì?",
    ],
}


class MedicalChatbot:
    def __init__(self):
        self.model = SentenceTransformer("keepitreal/vietnamese-sbert")
        self.index = faiss.read_index(str(BASE_DIR / "medical.index"))

        with open(BASE_DIR / "medical_metadata.pkl", "rb") as f:
            data = pickle.load(f)

        self.questions = data["questions"]
        self.diseases = data["diseases"]


    def search(self, query, k=3):
        query_vector = self.model.encode([query]).astype("float32")
        distances, indices = self.index.search(query_vector, k)

        context = []

        for i in indices[0]:
            context.append(
                f"Bệnh: {self.diseases[i]} - Triệu chứng: {self.questions[i]}"
            )

        return context

    def ask_llm(self, query, context, history=None):
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

    def _write_data_url_to_temp_file(self, data_url, file_name=None, mime_type=None):
        if not data_url:
            return None

        if "," not in data_url:
            raise ValueError("Invalid image data URL")

        header, encoded = data_url.split(",", 1)
        if not mime_type and header.startswith("data:") and ";base64" in header:
            mime_type = header[5: header.index(";base64")]

        suffix_map = {
            "image/png": ".png",
            "image/jpeg": ".jpg",
            "image/jpg": ".jpg",
            "image/webp": ".webp",
        }
        suffix = suffix_map.get(mime_type or "", "")
        if not suffix and file_name:
            suffix = Path(file_name).suffix or ".jpg"
        if not suffix:
            suffix = ".jpg"

        binary = base64.b64decode(encoded)
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        temp_file.write(binary)
        temp_file.flush()
        temp_file.close()
        return temp_file.name

    def analyze_image(self, image_path):
        from predict_skin import predict_image

        return predict_image(image_path)

    def _format_image_results(self, image_results):
        formatted = []
        for disease, score in image_results:
            formatted.append({
                "label": str(disease),
                "confidence": float(score),
            })
        return formatted

    def get_sample_prompts(self, mode="general"):
        prompts = SAMPLE_PROMPTS.get(mode) or SAMPLE_PROMPTS["general"]

        if len(self.questions) < 4:
            return prompts[:4]

        skin_keywords = ("da", "mụn", "ngứa", "nốt", "bong tróc", "phát ban", "mẩn")
        candidate_indices = []

        for index, question in enumerate(self.questions):
            normalized = str(question).lower()
            disease = str(self.diseases[index]).lower()
            is_skin_related = any(keyword in normalized or keyword in disease for keyword in skin_keywords)

            if mode == "dermatology" and is_skin_related:
                candidate_indices.append(index)
            elif mode != "dermatology" and not is_skin_related:
                candidate_indices.append(index)

        if len(candidate_indices) < 4:
            candidate_indices = list(range(len(self.questions)))

        random_indices = random.sample(candidate_indices, min(4, len(candidate_indices)))
        dataset_samples = [
            str(self.questions[index]).strip().replace("\n", " ")
            for index in random_indices
        ]
        return dataset_samples[:4]

    def suggest(self, mode="general"):
        return self.get_sample_prompts(mode)

    def answer(self, query, history=None, mode="general", image_data_url=None, image_file_name=None, image_mime_type=None):
        temp_image_path = None
        image_results = []
        image_predictions = []
        user_query = (query or "").strip()
        image_summary_text = ""

        try:
            if image_data_url:
                temp_image_path = self._write_data_url_to_temp_file(image_data_url, image_file_name, image_mime_type)
                try:
                    image_results = self.analyze_image(temp_image_path)
                    image_predictions = self._format_image_results(image_results)
                    image_summary_text = ", ".join([
                        f"{item['label']} ({round(item['confidence'] * 100, 2)}%)"
                        for item in image_predictions
                    ])
                except Exception as image_error:
                    user_query = user_query or "Tôi muốn được tư vấn ban đầu về tình trạng da trong ảnh."
                    user_query = f"{user_query}\n(Lưu ý: chưa phân tích được ảnh tự động — {image_error})"

            if image_predictions and not user_query:
                top = image_predictions[0]
                reply = (
                    "Kết quả nhận diện ảnh (chỉ mang tính tham khảo): "
                    f"khả năng cao nhất là {top['label']} ({round(top['confidence'] * 100, 2)}%). "
                    f"Top dự đoán: {image_summary_text}. "
                    "Bạn nên khám chuyên khoa da liễu để được chẩn đoán chính xác."
                )
                return {
                    "reply": reply,
                    "mode": mode,
                    "source": "image-classifier",
                    "image_predictions": image_predictions,
                    "suggestions": self.suggest(mode),
                }

            retrieval_query = user_query or "Tôi cần được tư vấn sức khỏe tổng quát."
            llm_query = retrieval_query
            if image_predictions:
                llm_query = (
                    f"{retrieval_query}\n"
                    f"Thông tin bổ sung từ mô hình nhận diện ảnh da: {image_summary_text}."
                )

            context = self.search(retrieval_query)
            try:
                reply = self.ask_llm(llm_query, context, history=history)
            except Exception as llm_error:
                top_context = context[0] if context else "chưa có dữ liệu tham chiếu phù hợp"
                reply = (
                    "Hiện tại tôi chưa thể kết nối mô hình ngôn ngữ. "
                    f"Dựa trên triệu chứng tương tự trong hệ thống: {top_context}. "
                    "Đây chỉ là định hướng ban đầu, bạn nên đi khám để được chẩn đoán chính xác."
                )
                return {
                    "reply": reply,
                    "mode": mode,
                    "source": "fallback",
                    "image_predictions": image_predictions,
                    "suggestions": self.suggest(mode),
                    "warning": str(llm_error),
                }
        finally:
            if temp_image_path:
                try:
                    Path(temp_image_path).unlink(missing_ok=True)
                except Exception:
                    pass

        return {
            "reply": reply,
            "mode": mode,
            "source": "python-service",
                "image_predictions": image_predictions,
            "suggestions": self.suggest(mode),
        }


def chatbot():

    assistant = MedicalChatbot()

    print("Medical AI Local Chatbot")
    print("------------------------")

    while True:

        query = input("\nBạn: ")

        if query == "exit":
            break

        context = assistant.search(query)

        answer = assistant.ask_llm(query, context)

        print("\nAI:", answer)


if __name__ == "__main__":
    chatbot()