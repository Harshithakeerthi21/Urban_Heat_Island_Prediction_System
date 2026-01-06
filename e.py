import streamlit as st
import pandas as pd
import base64
from io import BytesIO
from PIL import Image
import os
import matplotlib.cm as cm
import numpy as np
import torch
from torchvision import transforms
from segmentation_models_pytorch import Unet
import rasterio
import cv2
import matplotlib.pyplot as plt

# ------------------------- Streamlit Page Setup -------------------------
st.set_page_config(page_title="Urban Cooling Strategies", layout="wide")

st.markdown("""
    <style>
    html, body, .main {
        background: linear-gradient(135deg,#a3cef1 0%,#ffffff 100%);
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    h1 {
        color: #1e3c72;
        padding: 0.25em 0.5em;
        border-radius: 0.6rem;
        background: linear-gradient(90deg, #6dd5fa, #2980b9);
        box-shadow: 0 6px 12px -6px #2980b9;
    }
    .block-title {
        font-size: 1.4em;
        font-weight: 600;
        margin: 1.2em 0 0.3em 0;
        padding-left: 0.5em;
        border-left: 6px solid #2980b9;
        color: #094067;
    }
    .glass-card {
        background: rgba(255,255,255,0.85);
        border-radius: 1rem;
        box-shadow: 0 4px 22px 1px rgba(40,50,67,0.1);
        padding: 1em 1.2em;
        margin-bottom: 1em;
        color: #0a2540;
    }
    .stSelectbox label {
        color: #2980b9 !important;
        font-weight: 600;
    }
    .metric-label {
        font-weight: 700;
    }
    </style>
""", unsafe_allow_html=True)

# ------------------------- Helper Functions -------------------------
@st.cache_data
def load_data(csv_path):
    return pd.read_csv(csv_path)

def colorize_grayscale(img):
    arr = np.array(img)
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
    colored_arr = cm.get_cmap("RdYlGn")(arr)
    return Image.fromarray((colored_arr[:, :, :3] * 255).astype(np.uint8))

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@st.cache_resource
def load_model():
    model = Unet(
        encoder_name='resnet34',
        encoder_weights=None,
        in_channels=2,  # Two channels for NDVI+LST
        classes=1,
        activation=None
    )
    checkpoint = torch.load('uhi_unet_resnet34_oct12.pth', map_location=DEVICE)
    state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k
        if k.startswith('module.'):
            new_key = k[len('module.'):]
        elif k.startswith('model.'):
            new_key = k[len('model.'):]
        new_state_dict[new_key] = v
    model.load_state_dict(new_state_dict)
    model.to(DEVICE)
    model.eval()
    return model

def pad_to_divisible_32(arr):
    h, w = arr.shape[1], arr.shape[2]
    pad_h = (32 - h % 32) if h % 32 != 0 else 0
    pad_w = (32 - w % 32) if w % 32 != 0 else 0
    return np.pad(arr, ((0,0),(0,pad_h),(0,pad_w)), mode="reflect")

def normalize_ndvi(ndvi):
    ndvi = np.nan_to_num(ndvi, nan=0.0)
    return np.clip(ndvi, 0.0, 1.0).astype(np.float32)

def normalize_lst(lst):
    lst = np.nan_to_num(lst, nan=15.0)
    lst01 = (lst - 15.0) / (45.0 - 15.0)
    return np.clip(lst01, 0.0, 1.0).astype(np.float32)

def load_tiff_image_2ch(tiff_bytes, target_size=256):
    with rasterio.MemoryFile(tiff_bytes) as memfile:
        with memfile.open() as dataset:
            if dataset.count < 2:
                raise ValueError("TIFF must contain at least 2 bands (NDVI + LST).")

            arr = dataset.read()
            ndvi = normalize_ndvi(arr[0])
            lst = normalize_lst(arr[1])

            ndvi = cv2.resize(ndvi, (target_size, target_size))
            lst = cv2.resize(lst, (target_size, target_size))

            stacked = np.stack([ndvi, lst], axis=0)
            stacked = pad_to_divisible_32(stacked)
            tensor = torch.from_numpy(stacked).unsqueeze(0).float()
            return tensor

# ------------------------- Streamlit Main App -------------------------
def main():
    csv_path = r"C:\Users\kmadh\OneDrive\Desktop\major project\code3\citywise_uhi_with_ollama_strategies.csv"
    base_image_folder = r"C:\Users\kmadh\OneDrive\Desktop\major project\code3\UHI"
    df = load_data(csv_path)

    st.title("Urban Cooling Strategies Explorer")

    tab_selection, tab_strategy, tab_images, tab_metadata, tab_classifier = st.tabs(
        ["Selection", "Cooling Strategy", "Images", "Metadata", "UHI Classifier"]
    )

    # -------------------- Selection Tab --------------------
    with tab_selection:
        st.markdown('<div class="block-title">Select City</div>', unsafe_allow_html=True)
        city_selected = st.selectbox("City", sorted(df['city'].dropna().unique()))
        city_data = df[df['city'] == city_selected]
        row = city_data.iloc[0]

        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        c1.metric("Mean NDVI", f"{row.get('ndvi_mean', 'N/A'):.3f}")
        c2.metric("Mean LST", f"{row.get('lst_mean', 'N/A'):.2f} °C")
        c3.metric("UHI Fraction", f"{row.get('uhi_pred_fraction', 'N/A'):.3f}")
        st.markdown('</div>', unsafe_allow_html=True)

        csv_bytes = city_data.to_csv(index=False).encode()
        st.download_button("Download Metadata for selected city", csv_bytes, f"{city_selected}_metadata.csv", "text/csv")

    # -------------------- Strategy Tab --------------------
    with tab_strategy:
        uhi_class = row.get('uhi_class', '').strip().lower()
        if uhi_class == 'uhi':
            st.markdown('<div class="block-title">LLM-based Cooling Strategy</div>', unsafe_allow_html=True)
            strat_text = row['cooling_strategy']
            placeholder = st.empty()
            text_so_far = ""
            for char in str(strat_text):
                text_so_far += char
                placeholder.markdown(f"<div style='font-size:1.03em;color:#0a2540;'>{text_so_far}</div>", unsafe_allow_html=True)

            st.download_button("Download strategy as text file", strat_text, f"{city_selected}_strategy.txt")
        else:
            st.markdown('<div class="block-title">General Urban Environment Suggestions</div>', unsafe_allow_html=True)
            st.info(f"City: {city_selected} is classified as '{row.get('uhi_class', 'non-UHI')}'.")
            st.markdown("""
            <ul>
            <li>Maintain existing green spaces and plant more trees.</li>
            <li>Encourage sustainable urban planning practices.</li>
            <li>Monitor seasonal microclimate variations regularly.</li>
            <li>Promote public awareness about environmental conservation.</li>
            </ul>
            """, unsafe_allow_html=True)

    # -------------------- Images Tab --------------------
    with tab_images:
        st.markdown('<div class="block-title">City/UHI Image</div>', unsafe_allow_html=True)
        img_path_raw = row['image_path']
        if pd.notna(img_path_raw) and img_path_raw.strip():
            filename = os.path.basename(img_path_raw)
            local_img_path = os.path.join(base_image_folder, filename)
            if os.path.isfile(local_img_path):
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.image(local_img_path, caption=f"{city_selected}", use_column_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.info("No local city image found.")
        else:
            st.info("No city image info available.")

        st.markdown('<div class="block-title">NDVI Map (Red-Yellow-Green)</div>', unsafe_allow_html=True)
        if 'ndvi_image_base64' in row and pd.notna(row['ndvi_image_base64']):
            try:
                ndvi_bytes = base64.b64decode(row['ndvi_image_base64'])
                ndvi_img = Image.open(BytesIO(ndvi_bytes)).convert('L')
                colored_ndvi_img = colorize_grayscale(ndvi_img)
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.image(colored_ndvi_img, caption="NDVI Map", use_column_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            except Exception as e:
                st.warning(f"NDVI image could not be displayed: {e}")
        else:
            st.info("No NDVI image available for this record.")

    # -------------------- Metadata Tab --------------------
    with tab_metadata:
        st.markdown('<div class="block-title">Metadata Details</div>', unsafe_allow_html=True)
        meta_dict = row.to_dict()
        for k in ['cooling_strategy', 'ndvi_image_base64', 'image_path', 'year']:
            meta_dict.pop(k, None)
        cols = st.columns(min(len(meta_dict), 4))
        rows = (len(meta_dict) + 3) // 4
        i = 0
        for r in range(rows):
            for c in range(4):
                if i >= len(meta_dict):
                    break
                k, v = list(meta_dict.items())[i]
                with cols[c]:
                    st.markdown(
                        f"<div class='glass-card'><b>{k}</b><br><span style='color:#333533;'>{v}</span></div>",
                        unsafe_allow_html=True
                    )
                i += 1

    # -------------------- UHI Classifier Tab --------------------
    with tab_classifier:
        st.markdown('<div class="block-title">UHI Image Classification</div>', unsafe_allow_html=True)
        st.info("Upload a 2-band GeoTIFF (NDVI + LST). The model predicts Urban Heat Island probability.")

        uploaded_file = st.file_uploader(
            "Upload NDVI+LST 2-band TIFF file for UHI classification.",
            type=['tif', 'tiff']
        )

        if uploaded_file is not None:
            tiff_bytes = uploaded_file.read()
            try:
                model = load_model()
                tensor = load_tiff_image_2ch(tiff_bytes).to(DEVICE)

                with torch.no_grad():
                    output = model(tensor)
                    prob_map = torch.sigmoid(output).cpu().numpy()[0, 0]
                    uhi_fraction = float(prob_map.mean())
                    uhi_label = "UHI" if uhi_fraction >= 0.3 else "Non-UHI"

                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.markdown(f"### 🛰️ Prediction: **{uhi_label}**")
                st.metric("UHI Fraction", f"{uhi_fraction:.3f}")

                cmap_choice = "plasma" if uhi_label == "UHI" else "YlGn"
                fig, ax = plt.subplots(figsize=(4, 4))
                ax.imshow(prob_map, cmap=cmap_choice)
                ax.axis("off")
                st.pyplot(fig)
                st.markdown('</div>', unsafe_allow_html=True)

                # --- Cooling Strategies or General Suggestions ---
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                if uhi_label == "UHI":
                    st.markdown('<div class="block-title">Recommended Cooling Strategies</div>', unsafe_allow_html=True)
                    st.markdown("""
                    <ol style='font-size:1.05em; line-height:1.6em; color:#0a2540;'>
                        <li><b>Increase urban greenery</b> — introduce rooftop gardens, green walls, and urban forestry.</li>
                        <li><b>Adopt cool roofing materials</b> — use reflective, high-albedo coatings on rooftops and pavements.</li>
                        <li><b>Promote water-sensitive urban design</b> — integrate fountains, ponds, and permeable pavements for evaporative cooling.</li>
                        <li><b>Encourage mixed land use and compact city design</b> — reduce vehicle dependency and heat emissions.</li>
                        <li><b>Implement heat-aware urban planning policies</b> — prioritize zoning and construction materials that mitigate UHI effects.</li>
                    </ol>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown('<div class="block-title">Recommended Urban Environment Strategies</div>', unsafe_allow_html=True)
                    st.markdown("""
                    <ul style='font-size:1.05em; line-height:1.6em; color:#0a2540;'>
                        <li>Maintain existing green spaces and plant more trees.</li>
                        <li>Encourage sustainable urban design and rooftop greening.</li>
                        <li>Monitor microclimate data seasonally for early UHI detection.</li>
                        <li>Promote public awareness about sustainable cooling practices.</li>
                    </ul>
                    """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Error processing TIFF: {e}")

if __name__ == "__main__":
    main()
