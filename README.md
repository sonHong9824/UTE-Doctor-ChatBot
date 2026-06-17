# Redmine - Đặc tả chatbot y khoa UTE Doctor ChatBot

## 1. Thông tin chung

- Dự án: UTE-Doctor-ChatBot
- Tài liệu: Redmine đặc tả chức năng chatbot y khoa
- Phiên bản: 1.0
- Ngày cập nhật: 2026-06-18
- Phạm vi: Trả lời theo hội thoại văn bản, nhận diện ảnh da liễu, gợi ý hướng xử lý ban đầu, và trả lời tư vấn sức khỏe tổng quát

## 2. Mục tiêu nghiệp vụ

Mục tiêu của hệ thống là cung cấp một chatbot hỗ trợ y khoa mức tham khảo ban đầu cho người dùng tiếng Việt, gồm 2 luồng chính:

1. Trả lời câu hỏi sức khỏe dựa trên truy xuất tri thức từ bộ dữ liệu `ViMedical_Disease.csv`.
2. Phân tích ảnh da liễu bằng mô hình phân loại ảnh, sau đó giải thích bệnh được dự đoán cao nhất bằng ngôn ngữ dễ hiểu, kèm hướng xử lý an toàn ban đầu.

Hệ thống không thay thế bác sĩ và phải luôn nhấn mạnh đây chỉ là tư vấn tham khảo.

## 3. Bối cảnh hiện tại của hệ thống

Hệ thống đang được triển khai dưới dạng Python service tại cổng `8765`, sử dụng `ThreadingHTTPServer` và cung cấp các API chính:

- `GET /health`: kiểm tra trạng thái dịch vụ
- `GET /suggestions`: lấy các câu hỏi gợi ý theo chế độ
- `POST /chat`: nhận câu hỏi văn bản hoặc ảnh và trả lời

Luồng xử lý chính hiện tại:

1. Người dùng gửi message, history và/hoặc ảnh.
2. Nếu có ảnh, hệ thống lưu ảnh tạm, chạy mô hình nhận diện da liễu, lấy top prediction.
3. Hệ thống đưa top prediction vào LLM để sinh câu trả lời tự nhiên bằng tiếng Việt.
4. Nếu không có ảnh, hệ thống dùng embedding + FAISS để truy xuất ngữ cảnh từ bộ câu hỏi/bệnh trong `medical.index` và `medical_metadata.pkl`, sau đó gọi LLM.
5. Nếu LLM lỗi, hệ thống trả về fallback an toàn.

## 4. Thành phần hệ thống

### 4.1 Backend chatbot

File chính: `chatbot.py`

Chức năng:

- Nạp mô hình embedding `keepitreal/vietnamese-sbert`
- Nạp index FAISS từ `medical.index`
- Nạp metadata bệnh và câu hỏi từ `medical_metadata.pkl`
- Gọi `ollama.chat(model="llama3.2")` để sinh câu trả lời
- Xử lý upload ảnh qua data URL
- Chuẩn hóa nhãn ảnh sang tên tiếng Việt

### 4.2 HTTP service

File chính: `app.py`

Chức năng:

- Expose API `GET /health`
- Expose API `GET /suggestions`
- Expose API `POST /chat`
- Bắt lỗi client disconnect khi ghi response trên Windows

### 4.3 Mô hình ảnh

File chính: `predict_skin.py`

Chức năng:

- Đọc `skin_model.pth`
- Chạy suy luận top-k cho ảnh da
- Hỗ trợ TTA để tăng độ ổn định dự đoán

### 4.4 Dữ liệu tri thức

File chính: `ViMedical_Disease.csv`

Hiện trạng:

- Cột dữ liệu gồm `Disease` và `Question`
- Dùng để tạo bộ truy vấn ngữ cảnh cho chatbot text
- Dữ liệu hiện tại thiên về tri thức hỏi đáp theo triệu chứng, chưa có cột mô tả điều trị riêng

### 4.5 Mô hình ảnh da liễu

Dataset:

- `DermNet/train`
- `DermNet/test`

Các class ảnh hiện có gồm:

- Acne and Rosacea Photos
- Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions
- Atopic Dermatitis Photos
- Bullous Disease Photos
- Cellulitis Impetigo and other Bacterial Infections
- Eczema Photos
- Exanthems and Drug Eruptions
- Hair Loss Photos Alopecia and other Hair Diseases
- Herpes HPV and other STDs Photos
- Light Diseases and Disorders of Pigmentation
- Lupus and other Connective Tissue diseases
- Melanoma Skin Cancer Nevi and Moles
- Nail Fungus and other Nail Disease
- Poison Ivy Photos and other Contact Dermatitis
- Psoriasis pictures Lichen Planus and related diseases
- Scabies Lyme Disease and other Infestations and Bites
- Seborrheic Keratoses and other Benign Tumors
- Systemic Disease
- Tinea Ringworm Candidiasis and other Fungal Infections
- Urticaria Hives
- Vascular Tumors
- Vasculitis Photos
- Warts Molluscum and other Viral Infections

## 5. Luồng nghiệp vụ chi tiết

### 5.1 Luồng hội thoại văn bản

1. Người dùng nhập câu hỏi triệu chứng.
2. Hệ thống vector hóa câu hỏi bằng sentence transformer.
3. Hệ thống truy xuất top kết quả gần nhất trong FAISS index.
4. Hệ thống ghép context vào prompt cho LLM.
5. LLM trả về:
   - bệnh có khả năng nhất
   - giải thích
   - hướng điều trị
   - trả lời tự nhiên tiếng Việt

### 5.2 Luồng nhận diện ảnh

1. Người dùng gửi ảnh da.
2. Hệ thống giải mã data URL và lưu ảnh tạm.
3. Hệ thống gọi `predict_image()` để lấy top-k nhãn ảnh.
4. Hệ thống ánh xạ nhãn ảnh sang tên tiếng Việt chuẩn hóa.
5. Hệ thống tạo prompt cho LLM với yêu cầu:
   - giải thích bệnh có xác suất cao nhất
   - nói ngắn gọn triệu chứng thường gặp
   - hướng xử lý ban đầu an toàn
   - nhắc dấu hiệu cần khám sớm hoặc đi khám gấp
6. Hệ thống trả lời theo ngữ cảnh y khoa, không chỉ liệt kê tỷ lệ.

## 6. Yêu cầu chức năng

### 6.1 Tra cứu và trả lời câu hỏi sức khỏe

- Hệ thống phải tiếp nhận câu hỏi tiếng Việt.
- Hệ thống phải truy xuất được ngữ cảnh phù hợp từ `medical.index`.
- Hệ thống phải sinh câu trả lời tiếng Việt tự nhiên.
- Hệ thống phải trả lời theo hướng tham khảo, không khẳng định chẩn đoán tuyệt đối.

### 6.2 Xử lý ảnh da liễu

- Hệ thống phải nhận ảnh qua `imageDataUrl`.
- Hệ thống phải hỗ trợ tối thiểu các định dạng: PNG, JPG, WEBP.
- Hệ thống phải xuất top prediction và top-k predictions.
- Hệ thống phải giải thích bệnh top-1 thay vì chỉ trả bảng xác suất.
- Hệ thống phải cung cấp hướng xử lý ban đầu an toàn.

### 6.3 Ánh xạ nhãn ảnh sang tên tiếng Việt

- Hệ thống phải có bảng ánh xạ cho toàn bộ class ảnh DermNet.
- Hệ thống phải ưu tiên tên tiếng Việt dễ hiểu, gần đúng chuyên môn.
- Nếu không tìm thấy ánh xạ, hệ thống phải fallback về tên gốc của nhãn.

### 6.4 Gợi ý câu hỏi mẫu

- Hệ thống phải có danh sách gợi ý cho chế độ `general`.
- Hệ thống phải trả về danh sách gợi ý qua API `/suggestions`.

### 6.5 API dịch vụ

- `GET /health` trả trạng thái sống của service.
- `GET /suggestions?mode=general|dermatology` trả gợi ý theo chế độ.
- `POST /chat` nhận payload hội thoại và trả JSON response.

## 7. Yêu cầu phi chức năng

### 7.1 Hiệu năng

- Service phải phản hồi đủ nhanh cho hội thoại thông thường.
- Mô hình ảnh chỉ được load một lần trong process để giảm độ trễ.

### 7.2 Ổn định

- Service phải không crash khi client ngắt kết nối trong lúc ghi response.
- Khi LLM lỗi, hệ thống phải có fallback message an toàn.

### 7.3 Khả năng bảo trì

- Logic ảnh, logic text, và HTTP service phải tách tương đối rõ.
- Mapping label sang tiếng Việt phải dễ mở rộng.

### 7.4 Tính an toàn y tế

- Mọi câu trả lời phải có ngữ cảnh cảnh báo tham khảo.
- Không được đưa ra lời khuyên nguy hiểm như tự dùng thuốc mạnh khi chưa rõ chẩn đoán.

## 8. Đặc tả API

### 8.1 GET /health

Request:

- Không có body

Response 200:

```json
{
  "status": "ok",
  "service": "medical-assistant"
}
```

### 8.2 GET /suggestions

Query:

- `mode`: `general` hoặc `dermatology`

Response 200:

```json
{
  "mode": "general",
  "suggestions": ["...", "...", "...", "..."]
}
```

### 8.3 POST /chat

Request body:

```json
{
  "message": "...",
  "mode": "general",
  "history": [],
  "imageDataUrl": "data:image/jpeg;base64,...",
  "imageFileName": "sample.jpg",
  "imageMimeType": "image/jpeg"
}
```

Quy tắc:

- Bắt buộc có ít nhất `message` hoặc `imageDataUrl`
- Nếu có ảnh, hệ thống phải xử lý top prediction và trả thêm `image_predictions`

Response thành công chứa tối thiểu:

- `reply`
- `mode`
- `source`
- `suggestions`
- `image_predictions`

## 9. Tiêu chí nghiệm thu

### 9.1 Với câu hỏi văn bản

- Hệ thống trả lời được bằng tiếng Việt.
- Hệ thống có tham chiếu từ dữ liệu bệnh tương tự.
- Hệ thống không chỉ trả câu trả lời chung chung.

### 9.2 Với ảnh da liễu

- Hệ thống nhận được ảnh và trả về dự đoán top-1, top-k.
- Hệ thống giải thích bệnh top-1 bằng nội dung y khoa dễ hiểu.
- Hệ thống nêu cách xử lý ban đầu an toàn.
- Hệ thống nhắc đi khám da liễu nếu cần.
- Hệ thống không trả về chỉ danh sách phần trăm.

### 9.3 Với lỗi hệ thống

- Nếu LLM lỗi, hệ thống vẫn trả được fallback.
- Nếu ảnh lỗi phân tích, hệ thống vẫn trả được câu trả lời dựa trên text prompt.
- Nếu client ngắt kết nối, server vẫn giữ ổn định.

## 10. Test case đề xuất

### 10.1 Test hội thoại

1. Nhập câu hỏi triệu chứng tổng quát.
2. Kiểm tra chatbot có truy xuất context và trả lời tự nhiên.

### 10.2 Test ảnh da

1. Gửi ảnh có nhãn dự đoán cao như `Eczema Photos`.
2. Kiểm tra chatbot có giải thích bệnh chàm/viêm da cơ địa.
3. Kiểm tra có hướng dẫn xử lý ban đầu.

### 10.3 Test ảnh top-k

1. Gửi ảnh có nhiều dự đoán gần nhau.
2. Kiểm tra chatbot vẫn ưu tiên top-1 nhưng có nhắc top-k trong ngữ cảnh.

### 10.4 Test fallback

1. Tạm thời làm lỗi LLM hoặc ngắt kết nối Ollama.
2. Kiểm tra service vẫn trả fallback an toàn.

### 10.5 Test API

1. Gọi `/health`.
2. Gọi `/suggestions` theo từng mode.
3. Gọi `/chat` với message בלבד và với ảnh.

## 11. Rủi ro và lưu ý

- Kết quả nhận diện ảnh chỉ mang tính tham khảo.
- Một số class ảnh dễ bị nhầm lẫn lâm sàng, đặc biệt giữa chàm, viêm da cơ địa, viêm da tiếp xúc và nấm da.
- Dữ liệu `ViMedical_Disease.csv` hiện là bộ câu hỏi/triệu chứng, chưa phải bộ hướng dẫn điều trị chuyên sâu theo bệnh.
- Câu trả lời của LLM phụ thuộc chất lượng prompt và mô hình `llama3.2` đang chạy trong môi trường Ollama.

## 12. Đề xuất mở rộng

- Bổ sung bảng tri thức theo từng bệnh da để chatbot giải thích chi tiết và nhất quán hơn.
- Chuẩn hóa mapping giữa nhãn ảnh DermNet và tên bệnh tiếng Việt theo chuyên môn da liễu.
- Thêm chế độ phản hồi riêng cho ảnh, ví dụ:
  - mô tả bệnh
  - cách chăm sóc tại nhà
  - dấu hiệu nguy hiểm
  - khuyến nghị đi khám
- Bổ sung logging và tracing cho các lỗi LLM, lỗi ảnh và lỗi API.

## 13. Kết luận

Hệ thống hiện tại đã có đủ 3 lớp năng lực chính:

1. Trả lời y khoa bằng tri thức từ dữ liệu câu hỏi - bệnh.
2. Nhận diện và giải thích ảnh da liễu.
3. Phục vụ qua HTTP API để tích hợp vào frontend hoặc các hệ thống khác.

Tài liệu này mô tả theo góc nhìn BA để có thể dùng làm Redmine issue, mô tả task cho dev, và làm tiêu chí test cho QA.
