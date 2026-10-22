# app.py — Dashboard Streamlit Final (Lengkap 3 Tab)
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# KONFIGURASI HALAMAN & JUDUL
# ==========================================
st.set_page_config(page_title="Klasifikasi Biota Laut", page_icon="🌊", layout="wide")

st.title("🌊 Dashboard Klasifikasi Biota Laut")
st.markdown("""
Aplikasi ini menggunakan model **Convolutional Neural Network (CNN)** untuk mengklasifikasikan gambar biota laut ke dalam **23 kelas**. 
Dashboard ini menyajikan hasil eksperimen, evaluasi model, analisis visual, serta fitur prediksi secara *real-time*.
""")

# ==========================================
# DATA & KAMUS PENDUKUNG
# ==========================================
CLASS_NAMES = [
    'Clams', 'Corals', 'Crabs', 'Dolphin', 'Eel', 'Fish', 'Jelly Fish', 'Lobster', 
    'Nudibranchs', 'Octopus', 'Otter', 'Penguin', 'Puffers', 'Sea Rays', 
    'Sea Urchins', 'Seahorse', 'Seal', 'Sharks', 'Shrimp', 'Squid', 
    'Starfish', 'Turtle_Tortoise', 'Whale'
]

ANALISIS_BIOTA = {
    'Turtle_Tortoise': 'Penyu/Kura-kura. Reptil laut dengan cangkang keras yang berfungsi sebagai pelindung tubuh dari predator.',
    'Jelly Fish': 'Ubur-ubur. Hewan laut tak bertulang belakang dengan tubuh transparan seperti agar-agar dan memiliki tentakel berbisa.',
    'Dolphin': 'Lumba-lumba. Mamalia laut cerdas yang bernapas menggunakan paru-paru dan sering berkomunikasi dengan suara ultrasonik.',
    'Sharks': 'Hiu. Predator puncak di ekosistem laut dengan kerangka tulang rawan dan gigi tajam yang terus berganti.',
    'Sea Urchins': 'Bulu Babi. Hewan laut berduri yang bergerak lambat dan memakan alga di dasar laut.',
    'Whale': 'Paus. Mamalia laut terbesar di bumi yang bermigrasi jarak jauh dan menyaring makanannya menggunakan balin.',
    'Octopus': 'Gurita. Hewan moluska cerdas dengan delapan lengan yang dilengkapi alat pengisap untuk menangkap mangsa.',
    'Puffers': 'Ikan Buntal. Ikan yang dapat mengembangkan tubuhnya menjadi bola berduri saat merasa terancam.',
    'Sea Rays': 'Ikan Pari. Ikan bertulang rawan dengan tubuh pipih dan sirip dada besar yang menyerupai sayap.',
    'Corals': 'Koral. Hewan laut tak bertulang belakang yang membentuk terumbu karang dan bersimbiosis dengan alga.',
    'Nudibranchs': 'Nudibranch (Siput Laut). Moluska laut berwarna cerah yang tidak memiliki cangkang dan sering disebut kelinci laut.',
    'Otter': 'Berang-berang Laut. Mamalia laut yang sering terlihat mengambang di permukaan air sambil memecahkan kerang.',
    'Crabs': 'Kepiting. Krustasea dengan cangkang keras dan capit yang digunakan untuk pertahanan serta mencari makan.',
    'Lobster': 'Lobster. Krustasea besar dengan capit yang kuat dan hidup di celah-celah karang atau dasar laut.',
    'Starfish': 'Bintang Laut. Echinodermata dengan bentuk bintang bersudut lima dan memiliki kemampuan regenerasi lengan.',
    'Clams': 'Kerang. Moluska bercangkang dua (bivalvia) yang menyaring makanan dari air laut.',
    'Eel': 'Belut Laut. Ikan bertubuh panjang seperti ular yang sering bersembunyi di bebatuan atau pasir laut.',
    'Fish': 'Ikan. Hewan vertebrata air yang bernapas dengan insang dan bergerak menggunakan sirip.',
    'Shrimp': 'Udang. Krustasea kecil yang menjadi makanan penting bagi banyak hewan laut lainnya.',
    'Squid': 'Cumi-cumi. Moluska laut yang memiliki tentakel panjang dan dapat menyemprotkan tinta saat terancam.',
    'Penguin': 'Pinguin. Burung laut yang tidak bisa terbang di udara, namun sangat lihai berenang di air dingin.',
    'Seahorse': 'Kuda Laut. Ikan kecil dengan bentuk unik menyerupai kuda, di mana jantannya yang bertugas mengandung telur.',
    'Seal': 'Anjing Laut. Mamalia laut yang menghabiskan sebagian besar waktunya di air namun naik ke darat untuk beristirahat.'
}

DISTRIBUSI_DATA = {
    'Turtle_Tortoise': 1903, 'Jelly Fish': 845, 'Dolphin': 782, 'Sharks': 590,
    'Sea Urchins': 579, 'Whale': 572, 'Octopus': 562, 'Puffers': 531,
    'Sea Rays': 517, 'Corals': 500, 'Nudibranchs': 500, 'Otter': 500,
    'Crabs': 499, 'Lobster': 499, 'Starfish': 499, 'Clams': 497,
    'Eel': 497, 'Fish': 494, 'Shrimp': 488, 'Squid': 483, 'Penguin': 482,
    'Seahorse': 450, 'Seal': 450
}

@st.cache_resource
def load_model():
    model_path = 'models/cnn_ocean_model.keras'
    if os.path.exists(model_path):
        return tf.keras.models.load_model(model_path)
    return None

model = load_model()

# ==========================================
# SIDEBAR (RINGKASAN DATASET)
# ==========================================
st.sidebar.header("📊 Ringkasan Dataset")
st.sidebar.info(
    "- **Total Gambar:** 13.711\n"
    "- **Jumlah Kelas:** 23\n"
    "- **Proporsi Data Latih:** 80% (10.968 gambar)\n"
    "- **Proporsi Data Uji:** 20% (2.743 gambar)\n"
    "- **Arsitektur:** CNN (3 Layer Conv & MaxPooling)"
)

# ==========================================
# MEMBUAT 3 TABS
# ==========================================
tab1, tab2, tab3 = st.tabs([
    "📈 Hasil Evaluasi & Dataset", 
    "🔍 Uji Prediksi & Analisis",
    "🔬 Analisis Visual Model"
])

# ------------------------------------------
# TAB 1: GRAFIK DISTRIBUSI, HASIL TRAINING, EVALUASI
# ------------------------------------------
with tab1:
    st.header("1. Grafik Distribusi Data per Kelas")
    st.markdown("Dataset ini memiliki ketidakseimbangan kelas (*class imbalance*) yang cukup wajar. Kelas **Turtle_Tortoise** mendominasi data.")
    
    df_dist = pd.DataFrame(list(DISTRIBUSI_DATA.items()), columns=['Kelas', 'Jumlah Gambar'])
    df_dist = df_dist.sort_values(by='Jumlah Gambar', ascending=False)
    
    fig_dist, ax_dist = plt.subplots(figsize=(10, 5))
    sns.barplot(x='Jumlah Gambar', y='Kelas', data=df_dist, palette='viridis', ax=ax_dist)
    ax_dist.set_title('Distribusi Jumlah Gambar per Kelas')
    st.pyplot(fig_dist)
    
    st.divider()

    st.header("2. Hasil Pelatihan & Metrik Evaluasi")
    if not os.path.exists('outputs/grafik_training.png'):
        st.warning("⚠️ File evaluasi belum ada. Jalankan Cell 13 di Notebook Anda terlebih dahulu.")
    else:
        # Metrik Utama (Accuracy, Precision, Recall, F1)
        df_report = pd.read_csv('outputs/classification_report.csv', index_col=0)
        
        accuracy = df_report.loc['accuracy', 'f1-score']
        precision_avg = df_report.loc['weighted avg', 'precision']
        recall_avg = df_report.loc['weighted avg', 'recall']
        f1_avg = df_report.loc['weighted avg', 'f1-score']
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy", f"{accuracy*100:.2f}%")
        col2.metric("Precision (Avg)", f"{precision_avg*100:.2f}%")
        col3.metric("Recall (Avg)", f"{recall_avg*100:.2f}%")
        col4.metric("F1-Score (Avg)", f"{f1_avg*100:.2f}%")

        st.subheader("Grafik Akurasi & Loss Pelatihan")
        st.image('outputs/grafik_training.png', use_container_width=True)

        st.subheader("Confusion Matrix")
        st.image('outputs/confusion_matrix.png', use_container_width=True)

        st.subheader("Tabel Classification Report (Detail Per Kelas)")
        df_display = df_report.drop(['accuracy', 'macro avg', 'weighted avg'], errors='ignore')
        st.dataframe(df_display.style.background_gradient(cmap='Blues', subset=['precision', 'recall', 'f1-score']))

# ------------------------------------------
# TAB 2: UPLOAD GAMBAR & PREDIKSI REAL-TIME
# ------------------------------------------
with tab2:
    st.header("Uji Prediksi & Analisis Gambar Baru")
    uploaded_file = st.file_uploader("Pilih gambar (JPG, PNG, JPEG)...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Gambar yang Diunggah")
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Gambar Asli", use_container_width=True)
            
        with col2:
            st.subheader("Hasil Prediksi Model")
            
            if model is None:
                st.error("⚠️ Model tidak ditemukan! Pastikan file `cnn_ocean_model.keras` ada di folder `models/`.")
            else:
                with st.spinner('Menganalisis gambar...'):
                    # Preprocessing
                    img_resized = image.resize((96, 96))
                    img_array = np.array(img_resized)
                    img_array = np.expand_dims(img_array, axis=0)
                    
                    # Prediksi
                    predictions = model.predict(img_array)
                    score = tf.nn.softmax(predictions[0])
                    
                    predicted_class = CLASS_NAMES[np.argmax(score)]
                    confidence = np.max(score) * 100
                    
                    # Tampilkan Nama Kelas
                    st.success(f"**Prediksi: {predicted_class.replace('_', ' ')}**")
                    st.metric(label="Tingkat Keyakinan (Confidence)", value=f"{confidence:.2f}%")
                    
                    # Berikan Analisis
                    st.info(f"**Analisis Gambar:**\n{ANALISIS_BIOTA.get(predicted_class, 'Informasi belum tersedia.')}")
                    
                    # Top 3
                    st.write("**Top 3 Kemungkinan:**")
                    top_3_indices = np.argsort(score)[-3:][::-1]
                    for idx in top_3_indices:
                        st.write(f"- {CLASS_NAMES[idx].replace('_', ' ')}: {score[idx]*100:.2f}%")
                        st.progress(int(score[idx]*100))

# ------------------------------------------
# TAB 3: ANALISIS VISUAL MODEL (GRAD-CAM & SAMPEL)
# ------------------------------------------
with tab3:
    st.header("Analisis Visual Model CNN")
    st.markdown("Halaman ini menunjukkan bukti visual bagaimana model CNN mengambil keputusan.")
    
    st.subheader("1. Sampel Uji: Label Asli vs Prediksi Model")
    st.markdown("Gambar di bawah ini diambil langsung dari data uji (20% data yang tidak dilihat model saat pelatihan). Teks **hijau** berarti prediksi Benar, teks **merah** berarti prediksi Salah.")
    
    if os.path.exists('outputs/sample_prediksi.png'):
        st.image('outputs/sample_prediksi.png', use_container_width=True)
    else:
        st.warning("⚠️ File sample_prediksi.png belum ada. Jalankan Cell 16 di Notebook.")
        
    st.divider()
    
    st.subheader("2. Feature Importance (Grad-CAM)")
    st.markdown("""
    Visualisasi **Grad-CAM** (Gradient-weighted Class Activation Mapping) ini berfungsi untuk melihat area gambar yang paling memengaruhi model dalam memprediksi kelas tertentu. 
    *   **Area Biru:** Model kurang memperhatikan area ini.
    *   **Area Merah/Kuning:** Model sangat fokus pada area ini (misalnya sisik, sirip, atau cangkang) untuk menentukan klasifikasi.
    """)
    
    if os.path.exists('outputs/gradcam.png'):
        st.image('outputs/gradcam.png', use_container_width=True)
    else:
        st.warning("⚠️ File gradcam.png belum ada. Jalankan Cell 16 di Notebook.")

st.markdown("---")
st.caption("Tugas Praktikum Klasifikasi Citra CNN - Dashboard Interaktif")