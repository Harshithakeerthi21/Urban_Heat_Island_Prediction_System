import streamlit as st
import pandas as pd
import base64
from io import BytesIO
from PIL import Image
import os
import matplotlib.cm as cm
import numpy as np
import plotly.express as px

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

@st.cache_data
def load_data(csv_path):
    return pd.read_csv(csv_path)

def colorize_grayscale(img):
    arr = np.array(img)
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
    colored_arr = cm.get_cmap("RdYlGn")(arr)
    colored_img = Image.fromarray((colored_arr[:, :, :3] * 255).astype(np.uint8))
    return colored_img

def main():
    csv_path = r"C:\Users\kmadh\OneDrive\Desktop\major project\code3\citywise_uhi_with_ollama_strategies.csv"
    base_image_folder = r"C:\Users\kmadh\OneDrive\Desktop\major project\code3\UHI"
    df = load_data(csv_path)

    st.title("Urban Cooling Strategies Explorer")

    tab_selection, tab_strategy, tab_images, tab_metadata = st.tabs(
        ["Selection & Chart", "Cooling Strategy", "Images", "Metadata"]
    )

    with tab_selection:
        st.markdown('<div class="block-title">Select City and Year</div>', unsafe_allow_html=True)
        col1, col2 = st.columns([2,1])
        with col1:
            city_selected = st.selectbox("City", sorted(df['city'].dropna().unique()))
            city_data = df[df['city'] == city_selected]
        with col2:
            years = sorted(city_data['year'].dropna().unique())
            year_selected = st.selectbox("Year", years)
            row = city_data[city_data['year'] == year_selected].iloc[0]

        # Metrics summary
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        c1.metric("Mean NDVI", f"{row.get('ndvi_mean', 'N/A'):.3f}")
        c2.metric("Mean LST", f"{row.get('lst_mean', 'N/A'):.2f} °C")
        c3.metric("UHI Fraction", f"{row.get('uhi_pred_fraction', 'N/A'):.3f}")
        st.markdown('</div>', unsafe_allow_html=True)

        # NDVI over years chart
        sorted_years = city_data.sort_values('year')
        fig = px.line(
            sorted_years,
            x='year', y='ndvi_mean',
            markers=True,
            title=f"NDVI Variation in {city_selected} Over Years",
            labels={'ndvi_mean': 'NDVI Mean', 'year': 'Year'}
        )
        st.plotly_chart(fig, use_container_width=True)

        # Download metadata CSV
        csv_bytes = city_data.to_csv(index=False).encode()
        st.download_button("Download Metadata for selected city", csv_bytes, f"{city_selected}_metadata.csv", "text/csv")

    with tab_strategy:
        uhi_class = row.get('uhi_class', '').strip().lower()
        if uhi_class == 'uhi':
            st.markdown('<div class="block-title">LLM-based Cooling Strategy</div>', unsafe_allow_html=True)
            st.markdown(f"<div class='glass-card'><b>{city_selected}, {year_selected}</b><br>"
                        f"<span style='color:#555;'>Urban Cooling Recommendation:</span></div>", unsafe_allow_html=True)
            strat_text = row['cooling_strategy']
            placeholder = st.empty()
            text_so_far = ""
            for char in str(strat_text):
                text_so_far += char
                placeholder.markdown(f"<div style='font-size:1.03em;color:#0a2540;'>{text_so_far}</div>", unsafe_allow_html=True)

            # Download strategy text
            st.download_button("Download strategy as text file", strat_text, f"{city_selected}_{year_selected}_strategy.txt")
        else:
            st.markdown('<div class="block-title">General Urban Environment Suggestions</div>', unsafe_allow_html=True)
            st.info(f"City/year: {city_selected} ({year_selected}) is classified as '{row.get('uhi_class', 'non-UHI')}'.")
            st.markdown("""
            <ul>
            <li>Maintain existing green spaces and plant more trees to improve air quality.</li>
            <li>Encourage sustainable urban planning practices.</li>
            <li>Monitor seasonal microclimate variations regularly.</li>
            <li>Promote public awareness about environmental conservation.</li>
            <li>Even in areas without significant UHI, good urban health practices mutually benefit residents and ecosystems.</li>
            </ul>
            """, unsafe_allow_html=True)

    with tab_images:
        st.markdown('<div class="block-title">City/UHI Image</div>', unsafe_allow_html=True)
        img_path_raw = row['image_path']
        if pd.notna(img_path_raw) and img_path_raw.strip():
            filename = os.path.basename(img_path_raw)
            local_img_path = os.path.join(base_image_folder, filename)
            if os.path.isfile(local_img_path):
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.image(local_img_path, caption=f"{city_selected} {year_selected}", use_column_width=True)
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

    with tab_metadata:
        st.markdown('<div class="block-title">Metadata Details</div>', unsafe_allow_html=True)
        meta_dict = row.to_dict()
        meta_dict.pop('cooling_strategy', None)
        meta_dict.pop('ndvi_image_base64', None)
        meta_dict.pop('image_path', None)
        cols = st.columns(min(len(meta_dict), 4))  # max 4 cols per row
        rows = (len(meta_dict) + 3) // 4  # number of rows
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

if __name__ == "__main__":
    main()