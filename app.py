import streamlit as st
import joblib
from pyvi import ViTokenizer

# CẤU HÌNH NGƯỠNG TIN CẬY
THRESHOLD = 0.15  # Nếu độ tin cậy dưới 15% -> Trả lời là không biết

# Cấu hình trang web
st.set_page_config(
    page_title="Hỗ trợ sinh viên HUMG",
    page_icon="https://i.pinimg.com/originals/af/c0/8f/afc08f682827de2bc682d262448ade00.png",
    layout="wide"
)

# CSS tùy chỉnh
st.markdown(
    """ 
    <style>
    /* Chỉnh cỡ chữ cho nội dung chat */
    .stChatMessage .stMarkdown p, 
    .stChatMessage .stMarkdown li {
        font-size: 24px !important; 
        line-height: 1.6 !important;
    }
    /* Chỉnh cỡ chữ cho ô nhập liệu */
    .stChatInput textarea {
        font-size: 24px !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# 1. ĐỊNH NGHĨA CÁC HÀM XỬ LÝ (Giữ nguyên logic train)

vietnamese_stopwords = set([
    "là", "của", "các", "những", "cho", "với", "thì", "mà", "bị", "bởi", "được", "tại", "vì", "rằng", 
    "em", "muốn", "biết", "ạ", "ơi", "hỏi", "về", "như", "này", "khi", "trong", "để", "làm", "gì", 
    "nào", "ở", "có", "không", "nhưng", "rất", "cũng", "đã", "sẽ", "đang", "vẫn", "cứ", "chỉ", 
    "nếu", "hoặc", "hay", "do", "nên", "thôi", "nữa", "đâu", "đấy", "đây", "rồi", "chi", "nhiêu", 
    "để", "này", "kia"
])

def step1_lowercase(text):
    return text.lower()

def step2_tokenize(text):
    return ViTokenizer.tokenize(text)

def step3_remove_stopwords(text, stopwords=vietnamese_stopwords):
    words = text.split()
    filtered_words = [w for w in words if w not in stopwords]
    return " ".join(filtered_words)

def processing_pipeline(text):
    if not text: return ""
    text = step1_lowercase(text)
    text = step2_tokenize(text)
    text = step3_remove_stopwords(text)
    return text


# 2. LOAD MODEL ĐÃ LƯU

@st.cache_resource
def load_data():
    try:
        with open("chatbot_data.pkl", "rb") as f:
            data = joblib.load(f)
        return data["model"], data["vectorizer"], data["response_map"]
    except FileNotFoundError:
        st.error("Lỗi: Không tìm thấy file chatbot_data.pkl")
        return None, None, None

model, vectorizer, response_map = load_data()


# 3. GIAO DIỆN STREAMLIT 

st.markdown(
    """
    <h1 style='text-align: center; font-size: 50px; margin-bottom: 10px;'>
        Chatbot Hỗ Trợ Sinh Viên HUMG
    </h1>
    """,
    unsafe_allow_html=True
)

st.markdown(
    "<h4 style='text-align: center; font-weight: normal;'>Hỏi tôi bất cứ điều gì về <b>đăng ký môn, cách tính điểm, cách xếp hạng, điều kiện tốt nghiệp</b>...</h4>",
    unsafe_allow_html=True
)

st.markdown("---")

# Khởi tạo lịch sử chat
if "messages" not in st.session_state:
    st.session_state.messages = []

# Hiển thị lịch sử chat
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# 4. XỬ LÝ LOGIC TRẢ LỜI (CORE)

if prompt := st.chat_input("Nhập câu hỏi của bạn (Vui lòng nhập đầy đủ ý để đạt được kết quả tốt nhất. Xin cảm ơn!)"):
    if model is None:
        st.error("Lỗi: Chưa load được Model.")
    else:
        # Hiển thị câu hỏi người dùng
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        try:
            # a. Xử lý đầu vào
            processed_text = processing_pipeline(prompt)
            input_vec = vectorizer.transform([processed_text])
            
            # b. Tính xác suất 
            probs = model.predict_proba(input_vec)[0]  # Lấy danh sách xác suất
            max_prob = max(probs)                      # Lấy độ tin cậy cao nhất
            pred_index = probs.argmax()                # Lấy vị trí index
            pred_intent = model.classes_[pred_index]   # Lấy tên nhãn (Tag)

            # c. Kiểm tra ngưỡng tin cậy (Threshold)
            if max_prob > THRESHOLD:
                # Nếu tin cậy > 15% -> Tìm câu trả lời
                if pred_intent in response_map:
                    response_text = response_map[pred_intent][0]
                else:
                    response_text = "Xin lỗi, mình gặp lỗi dữ liệu (có nhãn nhưng thiếu câu trả lời)."
            else:
                # Nếu tin cậy <= 15% -> Trả lời không biết
                response_text = "Xin lỗi, câu này mình chưa được học hoặc bạn nhập chưa rõ nghĩa. Bạn thử diễn đạt lại xem?"
                
        except Exception as e:
            response_text = f"Đã xảy ra lỗi kỹ thuật: {str(e)}"

        # Hiển thị câu trả lời của Bot  
        with st.chat_message("assistant"):
            st.markdown(response_text)
        st.session_state.messages.append({"role": "assistant", "content": response_text})