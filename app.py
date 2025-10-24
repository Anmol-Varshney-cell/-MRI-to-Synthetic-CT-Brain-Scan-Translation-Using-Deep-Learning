import streamlit as st
import numpy as np
import tempfile
import os
from PIL import Image, ImageEnhance
import io

# -- Universal Loader for Medical & Image Files --
def load_uploaded_image(uploaded_file):
    filetype = os.path.splitext(uploaded_file.name)[-1].lower()
    if filetype in [".nii", ".nii.gz"]:
        tmp = tempfile.NamedTemporaryFile(suffix=filetype, delete=False)
        tmp.write(uploaded_file.read())
        tmp.close()
        img = nib.load(tmp.name)
        img_data = img.get_fdata()
        os.remove(tmp.name)
        return img_data
    elif filetype in [".png", ".jpg", ".jpeg"]:
        img = Image.open(uploaded_file).convert("L")
        img_data = np.array(img, dtype=np.float32)
        if img_data.ndim == 2:
            img_data = img_data[None, :, :]
        return img_data
    else:
        raise ValueError("Unsupported file type: %s" % filetype)

# THEME + FONT OPTIONS
THEMES = {
    "Default": {"bg": "#f0f2f6", "text": "#363636"},
    "Bright": {"bg": "#fff9db", "text": "#2b2a28"},
    "Dark": {"bg": "#000000", "text": "#f2f2f2"},
}

FONTS = {
    "Sans-serif": "font-family: Arial, Helvetica, sans-serif;",
    "Monospace": "font-family: 'Fira Mono', 'Menlo', 'Consolas', monospace;",
    "Serif": "font-family: Georgia, Cambria, 'Times New Roman', Times, serif;",
    "Cursive": "font-family: 'Comic Sans MS', 'Comic Sans', cursive, sans-serif;",
    "Fantasy": "font-family: Impact, fantasy;",
    "Modern": "font-family: 'Montserrat', Arial, Helvetica, sans-serif;",
    "Handwriting": "font-family: 'Brush Script MT', cursive;",
    "UI": "font-family: system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Oxygen', 'Ubuntu', 'Cantarell', 'Open Sans', 'Helvetica Neue', Arial, sans-serif;",
    "Rounded": "font-family: 'Varela Round', Arial, Helvetica, sans-serif;",
    "Slab": "font-family: 'Roboto Slab', Georgia, serif;",
}

st.set_page_config(page_title="MRI/CT/Segmentation Demo", page_icon="🧠", layout="centered")

# Appearance controls
theme = st.sidebar.selectbox("Theme", list(THEMES.keys()))
font = st.sidebar.selectbox("Font", list(FONTS.keys()))
st.markdown(
    f"""
    <style>
    body, .reportview-container .main, .stApp {{
        background-color: {THEMES[theme]['bg']} !important;
    }}
    div, p, label, span, h1, h2, h3, h4, h5, h6, .stButton>button {{
        color: {THEMES[theme]['text']} !important; {FONTS[font]}
    }}
    .stTextInput>div>div>input, .stTextArea>div>textarea, .stSelectbox>div>div>input {{
        background-color: {THEMES[theme]['bg']} !important;
        color: {THEMES[theme]['text']} !important;
    }}
    .stButton>button {{ background-color: #4278ff; color: white; border-radius: 8px; }}
    </style>
    """,
    unsafe_allow_html=True,
)

title_font = st.sidebar.selectbox("Title Font", ["Serif", "Cursive", "Sans-serif"])
FONT_MAP = {
    "Serif": "Georgia, Cambria, 'Times New Roman', Times, serif",
    "Cursive": "'Comic Sans MS', 'Comic Sans', cursive, sans-serif",
    "Sans-serif": "Arial, Helvetica, sans-serif",
}
st.markdown(
    f"""
    <style>
    h1 {{ 
        font-family: {FONT_MAP[title_font]} !important;
        font-weight: bold;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🧠 MRI-to-Synthetic CT Brain Scan Translation Using Deep Learning")

tab1, tab2, tab3, tab4 = st.tabs([
    "CT Generation", "Post-processing", "Batch Mode", "Segmentation"
])

# ---- TAB 1: CT Generation ----
with tab1:
    st.header("Single MRI → CT Demo")
    uploaded_file = st.file_uploader("Upload file (.nii/.nii.gz/.png/.jpg/.jpeg)", type=["nii","nii.gz","png","jpg","jpeg"], key="ctgen")
    if uploaded_file is not None:
        try:
            img_data = load_uploaded_image(uploaded_file)
            axis = st.radio("Axis", [0,1,2], key="ct_axis")
            n_slices = img_data.shape[axis]
            if n_slices == 1:
                slice_idx = 0
            else:
                slice_idx = st.slider("Slice", 0, n_slices-1, n_slices//2, key="ct_slice")
            if axis==0:
                slice_img = img_data[slice_idx,:,:]
            elif axis==1:
                slice_img = img_data[:,slice_idx,:]
            else:
                slice_img = img_data[:,:,slice_idx]
            norm_img = (slice_img-np.min(slice_img))/(np.max(slice_img)-np.min(slice_img))
            norm_img_uint8 = (norm_img*255).astype(np.uint8)
            st.image(norm_img_uint8, caption="MRI/Image slice", use_column_width=True)

            if st.button("Generate CT (demo)"):
                # Replace with YOUR CT model inference!
                ct_img = np.fliplr(norm_img_uint8)
                st.image(ct_img, caption="Synthetic CT (demo)", use_column_width=True)
                st.success("Synthetic CT generated!")
        except Exception as e:
            st.error(f"Couldn't load image: {e}")
    else:
        st.info("Upload a medical image file.")

# ---- TAB 2: Post-processing ----
with tab2:
    st.header("MRI Pre/Post-processing Demo")
    uploaded_file = st.file_uploader("Upload NIFTI or standard image for preprocessing", type=["nii","nii.gz","png","jpg","jpeg"], key="postproc")
    if uploaded_file is not None:
        try:
            img_data = load_uploaded_image(uploaded_file)
            axis = st.radio("Axis", [0,1,2], key="post_axis")
            n_slices = img_data.shape[axis]
            if n_slices == 1:
                slice_idx = 0
            else:
                slice_idx = st.slider("Slice", 0, n_slices-1, n_slices//2, key="post_slice")
            if axis==0:
                slice_img = img_data[slice_idx,:,:]
            elif axis==1:
                slice_img = img_data[:,slice_idx,:]
            else:
                slice_img = img_data[:,:,slice_idx]
            norm_img = (slice_img-np.min(slice_img))/(np.max(slice_img)-np.min(slice_img))
            norm_img_uint8 = (norm_img*255).astype(np.uint8)
            st.image(norm_img_uint8, caption="Original slice", use_column_width=True)

            # Contrast
            if st.button("Contrast Enhance"):
                value = st.slider("Contrast Level", 1.0, 2.5, 1.8)
                enhanced = Image.fromarray(norm_img_uint8).convert("L")
                enhanced = ImageEnhance.Contrast(enhanced).enhance(value)
                st.image(enhanced, caption="Contrast enhanced", use_column_width=True)
            # Smoothing
            if st.button("Gaussian Smoothing"):
                sigma = st.slider("Sigma", 0.1,5.0,1.0)
                smoothed = filters.gaussian(norm_img, sigma=sigma)
                st.image((smoothed*255).astype(np.uint8), caption="Smoothed", use_column_width=True)
        except Exception as e:
            st.error(f"Couldn't load image: {e}")
    else:
        st.info("Upload for preprocessing options.")

# ---- TAB 3: Batch Mode ----
with tab3:
    st.header("Batch MRI/Image Upload and Preview")
    uploaded_files = st.file_uploader("Upload multiple NIFTI or image files", type=["nii","nii.gz","png","jpg","jpeg"], key="batch", accept_multiple_files=True)
    if uploaded_files:
        for idx, file in enumerate(uploaded_files):
            try:
                img_data = load_uploaded_image(file)
                axis = 2
                n_slices = img_data.shape[axis]
                if n_slices == 1:
                    slice_idx = 0
                else:
                    slice_idx = st.slider(f"Slice {idx+1}", 0, n_slices-1, n_slices//2, key=f"batch_slice_{idx}")
                slice_img = img_data[:,:,slice_idx]
                norm_img = (slice_img-np.min(slice_img))/(np.max(slice_img)-np.min(slice_img))
                norm_img_uint8 = (norm_img*255).astype(np.uint8)
                st.image(norm_img_uint8, caption=f"Scan {idx+1} slice", use_column_width=True)
            except Exception as e:
                st.error(f"Batch file {idx+1}: {e}")
    else:
        st.info("Upload several files for batch preview.")

# ---- TAB 4: Segmentation (Demo) ----
with tab4:
    st.header("Segmentation Demo")
    uploaded_file = st.file_uploader("Upload MRI or image for segmentation", type=["nii","nii.gz","png","jpg","jpeg"], key="segm")
    if uploaded_file is not None:
        try:
            img_data = load_uploaded_image(uploaded_file)
            # --- Fix for 2-channel image ---
            if img_data.ndim == 3 and img_data.shape[2] == 2:
                img_data = img_data[..., 0][None, :, :]
            axis = st.radio("Axis", [0,1,2], key="seg_axis")
            n_slices = img_data.shape[axis]
            if n_slices == 1:
                slice_idx = 0
            else:
                slice_idx = st.slider("Slice", 0, n_slices-1, n_slices//2, key="seg_slice")
            if axis==0:
                slice_img = img_data[slice_idx,:,:]
            elif axis==1:
                slice_img = img_data[:,slice_idx,:]
            else:
                slice_img = img_data[:,:,slice_idx]
            norm_img = (slice_img-np.min(slice_img))/(np.max(slice_img)-np.min(slice_img))
            norm_img_uint8 = (norm_img*255).astype(np.uint8)
            st.image(norm_img_uint8, caption="MRI/Image Slice", use_column_width=True)

            if st.button("Run Segmentation (demo)"):
                mask = norm_img > 0.5
                mask = morphology.remove_small_objects(mask, min_size=64)
                st.image(mask.astype(float), caption="Segmentation Mask", use_column_width=True)
                overlay = np.stack([norm_img_uint8, mask.astype(np.uint8)*255], axis=-1)
                st.image(overlay, caption="Overlay (MRI + mask)", use_column_width=True)
        except Exception as e:
            st.error(f"Couldn't load image for segmentation: {e}")
    else:
        st.info("Upload a file to try segmentation.")

st.markdown("---")
st.caption("All demo features: extend with real models and more advanced processing as needed!\n© 2025 Creative MRI/CT App")