import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

# 設定頁面標題 (注意這裡都有引號)
st.set_page_config(page_title="學生證件照收集器", page_icon="📸")

def process_image(image_file):
    # 將上傳的檔案轉換為 OpenCV 格式
    file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    
    # 轉換為灰階以進行人臉偵測
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 載入人臉偵測模型 (Haar Cascade)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # 偵測人臉
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    if len(faces) == 0:
        return None, "無法偵測到人臉，請更換一張光線清晰、正面的照片。"
    
    # 取得最大的人臉 (假設畫面中最大的是主角)
    faces = sorted(faces, key=lambda x: x[2]*x[3], reverse=True)
    x, y, w, h = faces[0]
    
    # --- 計算 3:4 裁切範圍 ---
    img_h, img_w = img.shape[:2]
    
    # 臉部中心點
    center_x = x + w // 2
    center_y = y + h // 2
    
    # 設定裁切高度：以人臉高度的 2.5 倍為基準
    crop_h = int(h * 2.5)
    # 設定裁切寬度：高度的 3/4
    crop_w = int(crop_h * 3 / 4)
    
    # 計算左上角座標
    start_x = max(0, center_x - crop_w // 2)
    start_y = max(0, center_y - int(crop_h * 0.45))
    
    # 修正邊界
    end_x = min(img_w, start_x + crop_w)
    end_y = min(img_h, start_y + crop_h)
    
    # 再次確認比例
    if end_x - start_x < crop_w:
        start_x = max(0, end_x - crop_w)
    if end_y - start_y < crop_h:
        start_y = max(0, end_y - crop_h)
        
    # 執行裁切
    cropped_img = img[start_y:end_y, start_x:end_x]
    
    # 轉換回 RGB
    cropped_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)
    
    return cropped_img, "成功"

# --- 網頁介面設計 ---

st.title("📸 學生證件照電子檔收集器")
st.markdown("請輸入基本資料並上傳照片，系統將自動裁切為 **3:4 證件照比例**。")

col1, col2 = st.columns(2)

with col1:
    stu_class = st.text_input("班級", placeholder="例如：101")
    stu_seat = st.text_input("座號", placeholder="例如：01")

with col2:
    stu_name = st.text_input("姓名", placeholder="王小明")
    stu_id = st.text_input("學號", placeholder="112001")

uploaded_file = st.file_uploader("上傳照片 (支援 JPG, PNG)", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    if not (stu_class and stu_seat and stu_name and stu_id):
        st.warning("⚠️ 請先填寫完整的班級、座號、姓名與學號，才能處理照片。")
    else:
        with st.spinner('正在偵測人臉並裁切...'):
            try:
                processed_img, status = process_image(uploaded_file)
                
                if processed_img is not None:
                    st.success("✅ 照片處理完成！")
                    final_image = Image.fromarray(processed_img)
                    
                    cols = st.columns(2)
                    cols[0].image(uploaded_file, caption="原始照片", use_container_width=True)
                    cols[1].image(final_image, caption="自動裁切 (3:4)", use_container_width=True)
                    
                    buf = io.BytesIO()
                    final_image.save(buf, format="JPEG", quality=95)
                    byte_im = buf.getvalue()
                    
                    file_name = f"{stu_class}_{stu_seat}_{stu_name}_{stu_id}.jpg"
                    
                    st.download_button(
                        label="📥 下載處理後的照片",
                        data=byte_im,
                        file_name=file_name,
                        mime="image/jpeg"
                    )
                else:
                    st.error(f"❌ {status}")
            except Exception as e:
                st.error(f"發生錯誤：{str(e)}")

st.markdown("---")
st.caption("隱私聲明：此程式僅在本地端運作。")