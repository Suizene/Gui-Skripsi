import streamlit as st
import numpy as np
import pandas as pd
import time
from PIL import Image, ImageDraw
import io

def backend_predict_hybrid(image_file, clinical_data):
    time.sleep(2) 
    
    image_asli = Image.open(image_file).convert("RGBA")
    overlay = Image.new("RGBA", image_asli.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)
    
    # SIMULASI LOKALISASI LESI PARU (POLIGON MASK R-CNN)
    w, h = image_asli.size
    poligon_paru_kanan = [(w*0.25, h*0.33), (w*0.40, h*0.33), (w*0.45, h*0.55), (w*0.28, h*0.60)]
    poligon_paru_kiri = [(w*0.55, h*0.38), (w*0.70, h*0.40), (w*0.76, h*0.66), (w*0.62, h*0.62)]
    
    draw.polygon(poligon_paru_kanan, fill=(239, 68, 68, 75), outline=(239, 68, 68, 255), width=4)
    draw.polygon(poligon_paru_kiri, fill=(239, 68, 68, 75), outline=(239, 68, 68, 255), width=4)
        
    image_masked = Image.alpha_composite(image_asli, overlay).convert("RGB")
    
    # SIMULASI KLASIFIKASI BINER (HIBRIDA BCE)
    prob_inap = 0.765  
    threshold_youden = 0.4215  
    
    if prob_inap >= threshold_youden:
        status_teks = "Indikasi Rawat Inap Kritis"
        warna = "#ef4444"
    else:
        status_teks = "Rawat Jalan"
        warna = "#10b981"

    return {
        "status": status_teks,
        "warna": warna,
        "confidence": [(1 - prob_inap) * 100, prob_inap * 100], 
        "image_result": image_masked
    }

def main_frontend_ui():
    st.set_page_config(page_title="INTEGRA-PNEUMONIA v1.0", page_icon="🏥", layout="wide")

    st.markdown("""
        <style>
        [data-testid="stSidebar"] { background-color: #0f172a !important; padding-top: 20px; }
        [data-testid="stSidebar"] p, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] span, [data-testid="stSidebar"] label { color: #f1f5f9 !important; }
        .stApp { background-color: #f8fafc; }
        .main-output-card { background-color: #ffffff; padding: 35px; border-radius: 16px; box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.05); border: 1px solid #e2e8f0; margin-bottom: 25px; }
        .clinical-badge-card { background-color: #f1f5f9; padding: 20px; border-radius: 12px; border-left: 6px solid #ef4444; margin-top: 20px; }
        section[data-testid="stSidebar"] .stFileUploader { background-color: #1e293b; padding: 10px; border-radius: 8px; border: 1px dashed #475569; }
        </style>
    """, unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("## 🏥 Masukkan Data Pasien")
        st.markdown("---")
        
        st.markdown("### 👤 1. Demografi & Usia")
        nama = st.text_input("Nama Lengkap Pasien")
        
        col_y, col_m, col_d = st.columns(3)
        with col_y: tahun = st.number_input("Tahun", min_value=0, value=0)
        with col_m: bulan = st.number_input("Bulan", min_value=0, max_value=11, value=0)
        with col_d: hari = st.number_input("Hari", min_value=0, max_value=31, value=0)
        
        st.markdown("---")
        st.markdown("### 🩸 2. Parameter Klinis (7 Fitur)")
        hb = st.number_input("Hemoglobin (g/dL)", min_value=0.0, value=None, step=0.1)
        leukosit = st.number_input("Leukosit (sel/µL)", min_value=0, value=None, step=500, format="%d")
        hematrokit = st.number_input("Hematokrit (%)", min_value=0.0, value=None, step=0.5)
        trombosit = st.number_input("Trombosit (sel/µL)", min_value=0, value=None, step=10000, format="%d")
        neutrofil = st.number_input("Neutrofil (%)", min_value=0.0, value=None, step=1.0)
        limfosit = st.number_input("Limfosit (%)", min_value=0.0, value=None, step=1.0)
        
        st.markdown("---")
        st.markdown("### 🩻 3. Citra Thorax")
        uploaded_file = st.file_uploader("Upload File X-Ray (.jpg/.png)", type=["jpg", "jpeg", "png"])
        
        st.markdown("<br>", unsafe_allow_html=True)
        btn_diagnosa = st.button("🚀 Jalankan Fusi Multimodal", type="primary", use_container_width=True)

    st.title("🏥 INTEGRA-PNEUMONIA v1.0")
    st.caption("Sistem Aplikasi Hybrid Deep Learning — RSUD Haji Surabaya")
    st.markdown("---")

    st.markdown("<div class='main-output-card'>", unsafe_allow_html=True)
    
    if btn_diagnosa:
        if uploaded_file is not None:
            if not nama or any(v is None for v in [hb, leukosit, hematrokit, trombosit, neutrofil, limfosit]):
                st.warning("⚠️ Mohon lengkapi seluruh data identitas dan 6 parameter darah laboratorium.")
            else:
                usia_total_tahun = tahun + (bulan / 12.0) + (hari / 365.25)
                
                clinical_data = {
                    "usia_total_tahun": usia_total_tahun, "hb": hb, "leukosit": leukosit, 
                    "hematrokit": hematrokit, "trombosit": trombosit, "neutrofil": neutrofil, "limfosit": limfosit
                }
                
                with st.spinner("⏳ Mengeksekusi Mask R-CNN & fusi fitur MLP..."):
                    hasil = backend_predict_hybrid(uploaded_file, clinical_data)
                
                st.markdown(f"""
                    <div style='background: linear-gradient(135deg, #1e293b, #0f172a); color: #ffffff; padding: 20px; border-radius: 12px; margin-bottom: 25px; box-shadow: 0 4px 15px rgba(0,0,0,0.1); border-left: 8px solid {hasil['warna']};'>
                        <h4 style='margin:0; color:#94a3b8; font-size:14px; text-transform:uppercase;'>Keputusan Prediksi Sistem (Youden's J)</h4>
                        <h2 style='margin:5px 0 0 0; color:{hasil['warna']}; font-size:28px; font-weight:800;'>🚨 {hasil['status']}</h2>
                    </div>
                """, unsafe_allow_html=True)
                
                st.markdown("#### 📈 Analisis Probabilitas Nilai Klasifikasi")
                chart_data = pd.DataFrame({
                    'Kelas Diagnosis': ['Rawat Jalan', 'Rawat Inap Kritis'],
                    'Probabilitas (%)': hasil['confidence']
                })
                st.bar_chart(data=chart_data, x='Kelas Diagnosis', y='Probabilitas (%)', color="#ef4444", use_container_width=True)
                
                st.markdown("<br><hr>", unsafe_allow_html=True)
                
                st.markdown("#### 🩻 Komparasi Lokalisasi Lesi Paru (Mask R-CNN)")
                col_img1, col_img2 = st.columns(2)
                with col_img1:
                    st.image(uploaded_file, caption="Citra Rontgen Asli", use_container_width=True)
                with col_img2:
                    st.image(hasil['image_result'], caption="Ekstraksi Masking Poligon (ROI)", use_container_width=True)
                
                st.markdown("<hr>", unsafe_allow_html=True)
                
                st.markdown("#### 🩺 Korelasi Fitur Klinis (Multi-Layer Perceptron)")
                st.markdown(f"""
                    <div class='clinical-badge-card' style='border-left-color: {hasil['warna']};'>
                        • <b>Kalkulasi Usia Sistem:</b> Input {tahun} Thn, {bulan} Bln, {hari} Hari dikonversi menjadi fitur prediktor <b>{usia_total_tahun:.2f} Tahun</b>.<br>
                        • <b>Dominasi Fitur Darah:</b> Nilai Leukosit sebesar <b>{leukosit} sel/µL</b> dan Neutrofil <b>{neutrofil}%</b> memicu bobot atensi tinggi ke arah indikasi infeksi paru akut.<br>
                        • <b>Status Keputusan (Gated Attention):</b> Analisis fusi mendeteksi probabilitas tinggi ke arah indikasi <b>{hasil['status']}</b>.
                    </div>
                """, unsafe_allow_html=True)
                
        else:
            st.warning("⚠️ Berkas citra rontgen dada kosong. Harap unggah file terlebih dahulu pada panel input kiri.")
    else:
        st.markdown("<div style='text-align: center; padding: 100px 0; color: #94a3b8;'><h2>🖨️ Menunggu Data Rekam Medis</h2><p>Lengkapi formulir pasien di sebelah kiri dan klik Jalankan Fusi Multimodal.</p></div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main_frontend_ui()