# Chatbot Hỗ Trợ Sinh Viên HUMG (FAQ Quy chế Đào tạo)

#### Live Demo: https://chatbotfaq-dnphuc04.streamlit.app

Ứng dụng Chatbot thông minh giúp sinh viên **Trường Đại học Mỏ - Địa chất (HUMG)** tra cứu nhanh các thông tin về quy chế đào tạo, đăng ký môn học, cách tính điểm và điều kiện tốt nghiệp. Dự án sử dụng học máy (Machine Learning) để phân loại ý định người dùng.

---

## Tính năng chính
* **Nhận diện ý định (Intent Classification):** Hiểu được câu hỏi dù cách diễn đạt khác nhau.
* **Xử lý ngôn ngữ tự nhiên (NLP):** Sử dụng `Pyvi` tách từ tiếng Việt và loại bỏ stopwords.
* **Ngưỡng tin cậy (Confidence Threshold):** Chỉ trả lời khi độ tin cậy của mô hình đạt trên mức quy định (ví dụ > 20%).
* **Giao diện trực quan:** Tích hợp các nút gợi ý câu hỏi phổ biến (Quick Replies).

## Công nghệ sử dụng
* **Ngôn ngữ:** Python 3.x
* **Machine Learning:** `Scikit-learn` (N-gram & Naive Bayes).
* **Xử lý tiếng Việt:** `Pyvi` (ViTokenizer).
* **Giao diện Web:** `Streamlit`.
* **Lưu trữ mô hình:** `Joblib`.

## Cấu trúc thư mục dự án
```text
├── data/
│   └── data_daotao.json    # Dữ liệu huấn luyện (Intent, Patterns, Responses)
├── app.py                  # File mã nguồn chính chạy giao diện Streamlit
├── chatbot_data.pkl        # File nén chứa Model, Vectorizer và Map câu trả lời
├── requirements.txt        # Danh sách các thư viện cần thiết để deploy
└── README.md               # Tài liệu hướng dẫn dự án
```

## Cài đặt và Chạy thử (Local)
1. Clone repository:
```
git clone [https://github.com/dnphuc1211/ChatbotFAQ.git](https://github.com/dnphuc1211/ChatbotFAQ.git)
cd ChatbotFAQ
```
2. Cài đặt các thư viện cần thiết:
```
pip install -r requirements.txt
```
3. Khởi chạy Chatbot:
```
streamlit run app.py
```

## Pipeline của dự án:
1. Thu thập dữ liệu: Dữ liệu được cấu trúc dưới dạng JSON trong file data_daotao.json, bao gồm các nhãn ý định (tags), các mẫu câu hỏi (patterns) và câu trả lời tương ứng (responses).
2. Tiền xử lý: Chuyển về chữ thường, tách từ tiếng Việt bằng Pyvi và loại bỏ stopwords.
3. Vector hóa: Chuyển văn bản thành số bằng kỹ thuật Bigram (N-gram) để máy hiểu ngữ cảnh cặp từ.
4. Phân loại: Mô hình dự đoán xác suất (Naive Bayes) cho từng nhãn ý định.
5. Phản hồi: Nếu xác suất cao nhất vượt ngưỡng tin cậy, Bot lấy câu trả lời tương ứng để hiển thị.
6. Triển khai: Model sau khi huấn luyện được đóng gói vào chatbot_data.pkl để ứng dụng Streamlit gọi ra sử dụng ngay lập tức mà không cần huấn luyện lại mỗi khi chạy.
---
Người thực hiện: Đồng Ngọc Phúc

Email: dnphuc04@gmail.com

