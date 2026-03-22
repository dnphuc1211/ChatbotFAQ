import streamlit as st
import joblib
from pyvi import ViTokenizer

# --- 1. CẤU HÌNH TRANG & CSS ---
st.set_page_config(
    page_title="Hỗ trợ sinh viên HUMG",
    page_icon="https://i.pinimg.com/originals/af/c0/8f/afc08f682827de2bc682d262448ade00.png",
    layout="wide"
)

st.markdown(
    """ 
    <style>
    .stChatMessage .stMarkdown p, 
    .stChatMessage .stMarkdown li {
        font-size: 20px !important; 
        line-height: 1.6 !important;
    }
    .stChatInput textarea {
        font-size: 18px !important;
    }
    /* Chỉnh style cho các nút gợi ý */
    div.stButton > button {
        border-radius: 20px;
        border: 1px solid #ff4b4b;
        color: #ff4b4b;
        background-color: transparent;
        transition: all 0.3s;
    }
    div.stButton > button:hover {
        background-color: #ff4b4b;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- 2. TIỀN XỬ LÝ DỮ LIỆU ---
vietnamese_stopwords = set([
    "là", "của", "các", "những", "cho", "với", "thì", "mà", "bị", "bởi", "được", "tại", "vì", "rằng", 
    "em", "muốn", "biết", "ạ", "ơi", "hỏi", "về", "như", "này", "khi", "trong", "để", "làm", "gì", 
    "nào", "ở", "có", "không", "nhưng", "rất", "cũng", "đã", "sẽ", "đang", "vẫn", "cứ", "chỉ", 
    "nếu", "hoặc", "hay", "do", "nên", "thôi", "nữa", "đâu", "đấy", "đây", "rồi", "chi", "nhiêu", 
    "để", "này", "kia"
])

def processing_pipeline(text):
    if not text: return ""
    text = text.lower()
    text = ViTokenizer.tokenize(text)
    words = text.split()
    filtered_words = [w for w in words if w not in vietnamese_stopwords]
    return " ".join(filtered_words)

# --- 3. LOAD MODEL (SỬ DỤNG JOBLIB) ---
@st.cache_resource
def load_data():
    try:
        # Load file bằng joblib (nhanh và tối ưu hơn pickle cho model ML)
        data = joblib.load("chatbot_data.pkl")
        return data["model"], data["vectorizer"], data["response_map"]
    except Exception as e:
        st.error(f"Không tìm thấy dữ liệu mô hình: {e}")
        return None, None, None

model, vectorizer, response_map = load_data()
THRESHOLD = 0.20 # Ngưỡng tin cậy (có thể điều chỉnh)

# --- 4. GIAO DIỆN CHÍNH ---
st.markdown("<h1 style='text-align: center;'>Chatbot Hỗ Trợ Sinh Viên HUMG</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; font-weight: normal;'>Giải đáp Quy chế Đào tạo, Đăng ký môn học, Tốt nghiệp...</h4>", unsafe_allow_html=True)
st.markdown("---")

# Khởi tạo lịch sử chat
if "messages" not in st.session_state:
    st.session_state.messages = []

# Hiển thị các câu hỏi gợi ý (chỉ hiển thị khi chưa có hội thoại hoặc luôn hiển thị tùy ý)
st.write("**Câu hỏi thường gặp:**")
suggestions = [
    "Cách tính điểm trung bình học kỳ?",
    "Điều kiện xét tốt nghiệp?",
    "Đăng ký học cải thiện như thế nào?",
    "Quy định cảnh báo học vụ?"
]

cols = st.columns(len(suggestions))
clicked_prompt = None

for i, msg in enumerate(suggestions):
    if cols[i].button(msg, use_container_width=True):
        clicked_prompt = msg

# Hiển thị lịch sử chat từ session_state
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- 5. XỬ LÝ LOGIC CHAT ---
prompt = st.chat_input("Nhập câu hỏi tại đây...")

# Ưu tiên lấy câu hỏi từ nút bấm nếu có
if clicked_prompt:
    prompt = clicked_prompt

if prompt:
    # 1. Hiển thị câu hỏi người dùng
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 2. Xử lý câu trả lời
    if model is not None:
        try:
            processed_text = processing_pipeline(prompt)
            input_vec = vectorizer.transform([processed_text])
            
            # Dự đoán
            probs = model.predict_proba(input_vec)[0]
            max_prob = max(probs)
            pred_index = probs.argmax()
            pred_intent = model.classes_[pred_index]

            if max_prob > THRESHOLD:
                response_text = response_map.get(pred_intent, ["Xin lỗi, mình thiếu dữ liệu câu trả lời cho ý định này."])[0]
            else:
                response_text = "Xin lỗi, mình chưa hiểu ý bạn lắm. Bạn có thể đặt câu hỏi rõ ràng hơn về quy chế đào tạo không?"
        except Exception as e:
            response_text = f"Lỗi xử lý: {str(e)}"
    else:
        response_text = "Hệ thống đang bảo trì mô hình, vui lòng quay lại sau."

    # 3. Hiển thị câu trả lời của Assistant
    with st.chat_message("assistant"):
        st.markdown(response_text)
    st.session_state.messages.append({"role": "assistant", "content": response_text})

    # Nếu dùng nút bấm gợi ý, cần rerun để cập nhật giao diện chat mượt mà
    if clicked_prompt:
        st.rerun()