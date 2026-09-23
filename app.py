import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# 1. إعداد عنوان وتصميم الصفحة
st.set_page_config(page_title="مساعد مدرسة هباج عمار", page_icon="🏫", layout="centered")

st.markdown("<h1 style='text-align: center; color: #2C3E50;'>🏫 مدرسة هباج عمار - بني شبانة</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: #7F8C8D;'>المساعد الرقمي الذكي للمناهج والتشريع المدرسي</h3>", unsafe_allow_html=True)
st.write("---")

# 2. تحميل قاعدة البيانات التي قمت بحفظها مسبقاً (FAISS)
@st.cache_resource
def load_database():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    # تأكد من أن مجلد قاعدة البيانات موجود في نفس مسار ملف app.py
    vector_store = FAISS.load_local("habbaj_ammar_vector_db", embeddings, allow_dangerous_deserialization=True)
    return vector_store

with st.spinner("⏳ جاري تحميل ذاكرة المساعد وتجهيز التشريعات... يرجى الانتظار"):
    vector_store = load_database()

st.success("✅ النظام جاهز للاستقبال! اطرح سؤالك أدناه.")

# 3. صندوق إدخال السؤال من الأستاذ
user_question = st.text_input("📌 اكتب سؤالك هنا (مثال: ما هي خطوات فهم المنطوق؟ أو شروط رخص الغياب؟):")

if user_question:
    with st.spinner("🔍 جاري البحث في الوثائق الرسمية..."):
        # البحث عن أفضل 3 فقرات مطابقة
        docs = vector_store.similarity_search(user_question, k=3)
        
        st.markdown("### 🤖 الإجابة الموجهة للأستاذ:")
        for i, doc in enumerate(docs):
            st.info(f"**المرجع الرسمي {i+1}:**\n\n{doc.page_content}")
            
    st.markdown("---")
    st.markdown("💡 *ملاحظة: يُرجى الالتزام بالتوجيهات المستمدة من مناهج الجيل الثاني والتشريع المدرسي.*")
