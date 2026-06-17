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

    def _skin_label_to_vietnamese_name(self, label):
        normalized = str(label).strip().lower()

        label_map = {
            "acne and rosacea photos": "mụn trứng cá / rosacea",
            "actinic keratosis basal cell carcinoma and other malignant lesions": "dày sừng ánh sáng / ung thư da cần loại trừ",
            "atopic dermatitis photos": "viêm da cơ địa",
            "bullous disease photos": "bệnh da bóng nước",
            "cellulitis impetigo and other bacterial infections": "nhiễm khuẩn da",
            "eczema photos": "bệnh chàm / viêm da cơ địa",
            "exanthems and drug eruptions": "phát ban do thuốc / phát ban toàn thân",
            "hair loss photos alopecia and other hair diseases": "rụng tóc / bệnh tóc và da đầu",
            "herpes hpv and other stds photos": "nhiễm virus da niêm mạc",
            "light diseases and disorders of pigmentation": "rối loạn sắc tố da",
            "lupus and other connective tissue diseases": "lupus / bệnh mô liên kết",
            "melanoma skin cancer nevi and moles": "tổn thương sắc tố da cần loại trừ ung thư da",
            "nail fungus and other nail disease": "nấm móng / bệnh móng",
            "poison ivy photos and other contact dermatitis": "viêm da tiếp xúc / dị ứng da",
            "psoriasis pictures lichen planus and related diseases": "vảy nến / bệnh da viêm mạn tính",
            "scabies lyme disease and other infestations and bites": "ghẻ / nhiễm ký sinh trùng hoặc côn trùng đốt",
            "seborrheic keratoses and other benign tumors": "tổn thương da lành tính",
            "systemic disease": "biểu hiện da của bệnh lý toàn thân",
            "tinea ringworm candidiasis and other fungal infections": "nấm da",
            "urticaria hives": "mề đay",
            "vascular tumors": "u mạch máu / tổn thương mạch máu",
            "vasculitis photos": "viêm mạch máu",
            "warts molluscum and other viral infections": "mụn cóc / u mềm lây / nhiễm virus da",
        }

        if normalized in label_map:
            return label_map[normalized]

        cleaned = str(label).strip()
        cleaned = cleaned.replace(" Photos", "")
        cleaned = cleaned.replace(" pictures", "")
        return cleaned or str(label)


    def search(self, query, k=3):
        query_vector = self.model.encode([query]).astype("float32")
        distances, indices = self.index.search(query_vector, k)

        context = []

        for i in indices[0]:
            context.append(
                f"Bệnh: {self.diseases[i]} - Triệu chứng: {self.questions[i]}"
            )

        return context

    def ask_llm(self, query, context, history=None, instructions=None):
        context_text = "\n".join(context)

        extra_instructions = ""
        if instructions:
            extra_instructions = f"\n{instructions}\n"

        prompt = f"""
Bạn là bác sĩ AI.

Dựa vào dữ liệu sau:
{context_text}

Câu hỏi người dùng:
{query}

{extra_instructions}

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

    def _build_image_medical_query(self, top_prediction, image_predictions):
        top_label = str(top_prediction.get("label", "")).strip()
        top_name = self._skin_label_to_vietnamese_name(top_label)
        top_confidence = round(float(top_prediction.get("confidence", 0.0)) * 100, 2)
        top_predictions_text = ", ".join(
            f"{item['label']} ({round(item['confidence'] * 100, 2)}%)"
            for item in image_predictions
        )

        query = (
            f"Ảnh da gợi ý bệnh có khả năng cao nhất là {top_label} ({top_confidence}%). "
            f"Tên bệnh gần đúng để giải thích cho người dùng: {top_name}. "
            f"Các dự đoán khác: {top_predictions_text}."
        )

        instructions = (
            "Người dùng vừa gửi ảnh da. Hãy ưu tiên giải thích bệnh được dự đoán cao nhất bằng tiếng Việt dễ hiểu, "
            "nêu ngắn gọn vì sao bệnh này thường có biểu hiện như vậy, cách xử lý ban đầu an toàn tại nhà, "
            "khi nào cần khám da liễu sớm, và các dấu hiệu cần đi khám gấp. "
            "Không chỉ lặp lại tỷ lệ dự đoán. Nhắc rõ đây chỉ là gợi ý tham khảo, không phải chẩn đoán xác định."
        )

        return query, instructions, top_name

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

            retrieval_query = user_query or "Tôi cần được tư vấn sức khỏe tổng quát."
            llm_query = retrieval_query
            llm_instructions = None

            if image_predictions:
                top_prediction = image_predictions[0]
                image_query, llm_instructions, top_name = self._build_image_medical_query(top_prediction, image_predictions)
                llm_query = (
                    f"{retrieval_query}\n"
                    f"Thông tin bổ sung từ mô hình nhận diện ảnh da: {image_summary_text}.\n"
                    f"Bệnh gợi ý chính: {top_name}."
                )
                context_query = image_query
            else:
                context_query = retrieval_query

            context = self.search(context_query)
            try:
                reply = self.ask_llm(llm_query, context, history=history, instructions=llm_instructions)
            except Exception as llm_error:
                if image_predictions:
                    top_label = image_predictions[0]["label"]
                    top_name = self._skin_label_to_vietnamese_name(top_label)
                    reply = (
                        f"Ảnh gợi ý nhiều nhất đến {top_name}. "
                        "Bạn nên giữ vùng da sạch và khô, tránh gãi hoặc tự bôi thuốc mạnh khi chưa rõ chẩn đoán, "
                        "dùng dưỡng ẩm dịu nhẹ nếu da khô và đặt lịch khám da liễu để được xác nhận chính xác."
                    )
                else:
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
            "source": "image-llm" if image_predictions else "python-service",
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