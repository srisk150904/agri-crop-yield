# # import streamlit as st
# # import numpy as np
# # import tensorflow as tf
# # from PIL import Image

# # # Load trained model
# # # model = tf.keras.models.load_model("")
# # # Dummy model for frontend testing
# # class DummyModel:
# #     def predict(self, inputs):
# #         import numpy as np
# #         img_array, meta_array = inputs
# #         # Fake classification: always [0.3, 0.7] → "Paddy"
# #         class_pred = np.array([[0.3, 0.7]])
# #         # Fake regression: yield = area * 100 (just a mock formula)
# #         reg_pred = np.array([[meta_array[0][0] * 100]])
# #         return class_pred, reg_pred

# # model = DummyModel()


# # st.title("🌾 Crop Classification & Yield Prediction")
# # st.write("Upload a Landsat patch and enter metadata to classify crop type and predict yield.")

# # # --- Upload Landsat patch ---
# # uploaded_file = st.file_uploader("Upload Landsat Image (5-band PNG/JPG)", type=["png", "jpg", "jpeg"])

# # # --- Metadata input ---
# # st.subheader("Metadata Inputs")
# # area = st.number_input("Area (ha)", value=1.0, format="%.2f")
# # sowing = st.number_input("Sowing date (numeric/encoded)", value=100.0)
# # harvest = st.number_input("Harvest date (numeric/encoded)", value=200.0)
# # # Add more metadata fields as needed

# # if uploaded_file is not None:
# #     img = Image.open(uploaded_file).resize((12, 12))   # resize to match training
# #     st.image(img, caption="Uploaded Landsat Patch", use_column_width=True)

# #     img_array = np.array(img).astype("float32") / 255.0
# #     if img_array.ndim == 2:  # handle grayscale
# #         img_array = np.expand_dims(img_array, axis=-1)
# #     img_array = np.expand_dims(img_array, axis=0)  # (1,12,12,channels)

# #     meta_array = np.array([[area, sowing, harvest]])

# #     if st.button("🔍 Run Prediction"):
# #         # Run inference
# #         class_pred, reg_pred = model.predict([img_array, meta_array])
# #         class_label = int(np.argmax(class_pred, axis=1)[0])
# #         yield_pred = float(reg_pred[0][0])

# #         # Map labels (example: 0=Non-paddy, 1=Paddy)
# #         label_map = {0: "Non-Paddy", 1: "Paddy"}

# #         st.success(f"**Crop Type:** {label_map[class_label]}")
# #         st.info(f"**Predicted Yield:** {yield_pred:.2f} kg/ha")


# import streamlit as st
# import numpy as np
# import matplotlib.pyplot as plt

# # --- Dummy model for frontend testing ---
# class DummyModel:
#     def predict(self, inputs):
#         img_array, meta_array = inputs
#         # Fake classification: always "Paddy" with 70% confidence
#         class_pred = np.array([[0.3, 0.7]])
#         # Fake regression: yield = area * 100 (mock formula)
#         reg_pred = np.array([[meta_array[0][0] * 100]])
#         return class_pred, reg_pred

# # Swap this later with your real model
# # model = tf.keras.models.load_model("cropnet_model.h5")
# model = DummyModel()

# # --- Streamlit UI ---
# st.title("🌾 Crop Classification & Yield Prediction")
# st.write("Upload a Landsat patch (.npz) and enter metadata to classify crop type and predict yield.")

# # --- File uploader ---
# uploaded_file = st.file_uploader("Upload Landsat Patch (.npz)", type=["npz"])

# # --- Metadata input ---
# st.subheader("Metadata Inputs")
# area = st.number_input("Area (ha)", value=1.0, format="%.2f")
# sowing = st.number_input("Sowing date (numeric/encoded)", value=100.0)
# harvest = st.number_input("Harvest date (numeric/encoded)", value=200.0)

# if uploaded_file is not None:
#     with np.load(uploaded_file) as data:
#         # Load first array (usually stored as 'arr_0')
#         img_array = data[list(data.keys())[0]]

#     st.write("✅ Landsat patch loaded:", img_array.shape)

#     # Normalize (Landsat reflectance often 0–10000)
#     img_array = img_array.astype("float32")
#     if img_array.max() > 1.0:
#         img_array = img_array / 10000.0  

#     # Ensure shape = (H, W, C)
#     if img_array.shape[0] < img_array.shape[-1]:
#         img_array = np.transpose(img_array, (1, 2, 0))

#     # Show RGB preview (bands 3,2,1 as RGB if available)
#     if img_array.shape[-1] >= 3:
#         rgb_img = img_array[:, :, [2, 1, 0]]
#         rgb_img = (rgb_img - rgb_img.min()) / (rgb_img.max() - rgb_img.min() + 1e-6)
#         st.subheader("RGB Preview")
#         st.image(rgb_img, caption="Landsat Patch (RGB)", use_column_width=True)

#     # Expand dims for model
#     img_array_batch = np.expand_dims(img_array, axis=0)
#     meta_array = np.array([[area, sowing, harvest]])

#     # --- Prediction button ---
#     if st.button("🔍 Run Prediction"):
#         class_pred, reg_pred = model.predict([img_array_batch, meta_array])
#         class_label = int(np.argmax(class_pred, axis=1)[0])
#         yield_pred = float(reg_pred[0][0])

#         # Map labels (example: 0=Non-paddy, 1=Paddy)
#         label_map = {0: "Non-Paddy", 1: "Paddy"}

#         st.success(f"**Crop Type:** {label_map[class_label]}")
#         st.info(f"**Predicted Yield:** {yield_pred:.2f} kg/ha")
# ------------------------------------------------------------------------------------------------------------------------------------

# import streamlit as st
# import numpy as np

# # --- Dummy model for frontend testing ---
# class DummyModel:
#     def predict(self, inputs):
#         img_array, meta_array = inputs
#         # Fake classification: always "Paddy" with 70% confidence
#         class_pred = np.array([[0.3, 0.7]])
#         # Fake regression: yield = area * 100 (mock formula)
#         reg_pred = np.array([[meta_array[0][0] * 100]])
#         return class_pred, reg_pred

# # Swap with your real model later
# # model = tf.keras.models.load_model("cropnet_model.h5")
# model = DummyModel()

# # --- Streamlit UI ---
# st.title("🌾 Crop Classification & Yield Prediction")
# st.write("Upload a Landsat patch (.npz) with 5 bands and enter metadata to classify crop type and predict yield.")

# # --- File uploader ---
# uploaded_file = st.file_uploader("Upload Landsat Patch (.npz)", type=["npz"])

# # --- Metadata input ---
# st.subheader("Metadata Inputs")

# area = st.number_input("Area (ha)", value=1.0, format="%.2f", key="area_input")
# sowing_month = st.number_input("Sowing Month (numeric/encoded)", value=100.0, key="sowing_month_input")
# harvest_month = st.number_input("Harvest Month (numeric/encoded)", value=200.0, key="harvest_month_input")
# sowing_to_transplanting_days = st.number_input(
#     "Sowing to Transplanting Days (numeric/encoded)", value=200.0, key="sowing_to_transplanting_input"
# )

# if uploaded_file is not None:
#     with np.load(uploaded_file) as data:
#         # Load first array (usually stored as 'arr_0')
#         img_array = data[list(data.keys())[0]]

#     st.write("✅ Landsat patch loaded:", img_array.shape)

#     # Normalize (Landsat reflectance often 0–10000)
#     img_array = img_array.astype("float32")
#     if img_array.max() > 1.0:
#         img_array = img_array / 10000.0  

#     # Ensure shape = (H, W, C)
#     if img_array.ndim == 2:
#         img_array = np.expand_dims(img_array, axis=-1)
#     elif img_array.shape[0] < img_array.shape[-1]:
#         img_array = np.transpose(img_array, (1, 2, 0))

#     # --- Pseudo-RGB Preview (using SR_B4, SR_B5, SR_B6) ---
#     if img_array.shape[-1] >= 3:
#         pseudo_rgb = np.stack([
#             img_array[:, :, 0],  # SR_B4 (Red)
#             img_array[:, :, 1],  # SR_B5 (NIR)
#             img_array[:, :, 2],  # SR_B6 (SWIR1)
#         ], axis=-1)
#         pseudo_rgb = (pseudo_rgb - pseudo_rgb.min()) / (pseudo_rgb.max() - pseudo_rgb.min() + 1e-6)
#         st.subheader("Pseudo-RGB Preview (SR_B4, SR_B5, SR_B6)")
#         st.image(pseudo_rgb, caption="Landsat Patch (Pseudo-RGB)", use_column_width=True)

#     # Expand dims for model
#     img_array_batch = np.expand_dims(img_array, axis=0)   # (1, H, W, 5)
#     meta_array = np.array([[area, sowing, harvest]])

#     # --- Prediction button ---
#     if st.button("🔍 Run Prediction"):
#         class_pred, reg_pred = model.predict([img_array_batch, meta_array])
#         class_label = int(np.argmax(class_pred, axis=1)[0])
#         yield_pred = float(reg_pred[0][0])

#         # Map labels (example: 0=Non-paddy, 1=Paddy)
#         label_map = {0: "Non-Paddy", 1: "Paddy"}

#         st.success(f"**Crop Type:** {label_map[class_label]}")
#         st.info(f"**Predicted Yield:** {yield_pred:.2f} kg/ha")



# import streamlit as st
# import numpy as np
# import joblib
# import tensorflow as tf
# from sklearn.preprocessing import PowerTransformer

# # ======================================
# # --- Utility functions ---
# # ======================================
# def preprocess_landsat_image(data, target_bands=['SR_B4', 'SR_B5', 'SR_B6', 'ST_B10', 'ST_TRAD']):
#     """Preprocess Landsat .npz image for CNN (normalize, pad, stack 5 bands)."""
#     img_stack = []
#     raw_b4, raw_b5 = None, None
#     for band in target_bands:
#         if band in data:
#             band_img = data[band].astype(np.float32)
#             h, w = band_img.shape
#             pad_h, pad_w = max(0, 12 - h), max(0, 12 - w)
#             band_img = np.pad(band_img, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=0)
#             if band == "SR_B4": raw_b4 = band_img.copy()
#             if band == "SR_B5": raw_b5 = band_img.copy()
#             mean, std = np.mean(band_img), np.std(band_img)
#             band_img = (band_img - mean) / std if std > 0 else np.zeros_like(band_img)
#             img_stack.append(band_img)

#     if len(img_stack) != len(target_bands):
#         raise ValueError("Missing one or more Landsat bands.")
#     img = np.stack(img_stack, axis=-1)
#     ndvi = (raw_b5 - raw_b4) / (raw_b5 + raw_b4 + 1e-6)
#     ndvi_mean = float(np.nanmean(ndvi))
#     return img, ndvi_mean

# def compute_sentinel_features(sentinel_data):
#     """Compute VV_mean, VH_mean, VH_VV_ratio, and transformed ratio."""
#     VV = sentinel_data.get("VV")
#     VH = sentinel_data.get("VH")
#     if VV is None or VH is None:
#         raise ValueError("VV/VH bands missing in Sentinel data.")

#     VV_mean = float(np.nanmean(VV))
#     VH_mean = float(np.nanmean(VH))
#     VH_VV_ratio = float(np.nanmean(VH / (VV + 1e-6)))

#     pt = PowerTransformer(method="yeo-johnson", standardize=False)
#     VH_VV_ratio_trans2 = float(pt.fit_transform([[VH_VV_ratio]])[0][0])
#     return VV_mean, VH_mean, VH_VV_ratio, VH_VV_ratio_trans2

# # ======================================
# # --- Streamlit App ---
# # ======================================
# st.title("🌾 Hybrid CNN + LightGBM Crop Yield Prediction")
# st.write("Upload **Landsat** and **Sentinel** patches along with metadata to predict yield using your trained models.")

# # --- Model Uploaders ---
# st.sidebar.header("📦 Upload Models")
# cnn_model_file = st.sidebar.file_uploader("Upload CNN Model (.h5)", type=["h5"])
# lgbm_model_file = st.sidebar.file_uploader("Upload LightGBM Model (.pkl)", type=["pkl", "joblib"])

# # --- Input Uploads ---
# st.subheader("🌍 Upload Input Data")
# landsat_file = st.file_uploader("Upload Landsat Patch (.npz)", type=["npz"])
# sentinel_file = st.file_uploader("Upload Sentinel Patch (.npz)", type=["npz"])

# # --- Metadata Inputs ---
# st.subheader("📋 Metadata Inputs")
# area = st.number_input("Area (ha)", value=1.0, format="%.2f")
# sow_mon = st.number_input("Sowing Month (numeric/encoded)", value=6.0)
# har_mon = st.number_input("Harvest Month (numeric/encoded)", value=12.0)
# sow_to_trans_days = st.number_input("Sowing to Transplanting Days", value=25.0)
# trans_to_har_days = st.number_input("Transplanting to Harvest Days", value=100.0)

# # ======================================
# # --- Model Loading ---
# # ======================================
# cnn_model, lgbm_model = None, None
# if cnn_model_file is not None:
#     cnn_model = tf.keras.models.load_model(cnn_model_file)
#     st.sidebar.success("✅ CNN model loaded")

# if lgbm_model_file is not None:
#     lgbm_model = joblib.load(lgbm_model_file)
#     st.sidebar.success("✅ LightGBM model loaded")

# # ======================================
# # --- Run Prediction ---
# # ======================================
# if st.button("🔍 Run Prediction"):
#     if not (landsat_file and sentinel_file and cnn_model and lgbm_model):
#         st.error("❌ Please upload both images and both models.")
#         st.stop()

#     # --- Load and preprocess Landsat ---
#     with np.load(landsat_file) as ldata:
#         landsat_img, ndvi_val = preprocess_landsat_image(ldata)

#     # --- CNN Feature Extraction ---
#     img_input = np.expand_dims(landsat_img, axis=0)  # shape (1, 12, 12, 5)
#     cnn_features = cnn_model.predict(img_input)
#     cnn_features = cnn_features.flatten()  # 1D feature vector

#     # --- Sentinel feature extraction ---
#     with np.load(sentinel_file) as sdata:
#         VV_mean, VH_mean, VH_VV_ratio, VH_VV_ratio_trans2 = compute_sentinel_features(sdata)

#     # --- Apply transformations ---
#     sow_to_trans_log = np.log1p(sow_to_trans_days).astype(np.float64)

#     # --- Prepare final LightGBM input ---
#     tabular_features = np.array([
#         area, sow_mon, har_mon, sow_to_trans_log, trans_to_har_days,
#         VV_mean, VH_mean, VH_VV_ratio, VH_VV_ratio_trans2, ndvi_val
#     ])
#     full_input = np.concatenate([tabular_features, cnn_features])
#     full_input = full_input.reshape(1, -1)

#     # --- Predict yield ---
#     yield_pred_log = float(lgbm_model.predict(full_input)[0])
#     yield_pred = np.expm1(yield_pred_log)

#     # --- Display results ---
#     st.success(f"**Predicted Yield:** {yield_pred:.2f} kg/ha 🌾")
#     st.write("### 📊 Feature Summary")
#     st.dataframe({
#         "AREA": [area],
#         "sow_mon": [sow_mon],
#         "har_mon": [har_mon],
#         "sow_to_trans_days (log1p)": [sow_to_trans_log],
#         "trans_to_har_days": [trans_to_har_days],
#         "VV_mean": [VV_mean],
#         "VH_mean": [VH_mean],
#         "VH_VV_ratio": [VH_VV_ratio],
#         "VH_VV_ratio_trans2": [VH_VV_ratio_trans2],
#         "NDVI": [ndvi_val],
#         "CNN_features_dim": [cnn_features.shape[0]]
#     })



import streamlit as st
import numpy as np
import joblib
import tensorflow as tf
import tempfile
from sklearn.preprocessing import PowerTransformer

# ======================================
# --- Utility functions ---
# ======================================
def preprocess_landsat_image(data, target_bands=['SR_B4', 'SR_B5', 'SR_B6', 'ST_B10', 'ST_TRAD']):
    """Preprocess Landsat .npz image for CNN (normalize, pad, stack 5 bands)."""
    img_stack = []
    raw_b4, raw_b5 = None, None

    for band in target_bands:
        if band in data:
            band_img = data[band].astype(np.float32)
            h, w = band_img.shape
            pad_h, pad_w = max(0, 12 - h), max(0, 12 - w)
            band_img = np.pad(band_img, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=0)

            # Keep raw B4 and B5 for NDVI computation
            if band == "SR_B4":
                raw_b4 = band_img.copy()
            if band == "SR_B5":
                raw_b5 = band_img.copy()

            # Normalize each band
            mean, std = np.mean(band_img), np.std(band_img)
            band_img = (band_img - mean) / std if std > 0 else np.zeros_like(band_img)
            img_stack.append(band_img)

    if len(img_stack) != len(target_bands):
        raise ValueError("Missing one or more Landsat bands.")

    # Stack 5 normalized bands
    img = np.stack(img_stack, axis=-1)

    # Compute NDVI
    ndvi = (raw_b5 - raw_b4) / (raw_b5 + raw_b4 + 1e-6)
    ndvi_mean = float(np.nanmean(ndvi))
    return img, ndvi_mean


def compute_sentinel_features(sentinel_data):
    """Compute VV_mean, VH_mean, VH_VV_ratio, and transformed ratio."""
    VV = sentinel_data.get("VV")
    VH = sentinel_data.get("VH")
    if VV is None or VH is None:
        raise ValueError("VV/VH bands missing in Sentinel data.")

    VV_mean = float(np.nanmean(VV))
    VH_mean = float(np.nanmean(VH))
    VH_VV_ratio = float(np.nanmean(VH / (VV + 1e-6)))

    pt = PowerTransformer(method="yeo-johnson", standardize=False)
    VH_VV_ratio_trans2 = float(pt.fit_transform([[VH_VV_ratio]])[0][0])

    return VV_mean, VH_mean, VH_VV_ratio, VH_VV_ratio_trans2


# ======================================
# --- Streamlit App ---
# ======================================
st.title("🌾 Hybrid CNN + LightGBM Crop Yield Prediction")
st.write("Upload **Landsat** and **Sentinel** patches along with metadata to predict crop yield using your trained models.")

# --- Model Uploaders ---
st.sidebar.header("📦 Upload Models")
cnn_model_file = st.sidebar.file_uploader("Upload CNN Model (.h5)", type=["h5"])
lgbm_model_file = st.sidebar.file_uploader("Upload LightGBM Model (.pkl)", type=["pkl", "joblib"])

# --- Input Uploads ---
st.subheader("🌍 Upload Input Data")
landsat_file = st.file_uploader("Upload Landsat Patch (.npz)", type=["npz"])
sentinel_file = st.file_uploader("Upload Sentinel Patch (.npz)", type=["npz"])

# --- Metadata Inputs ---
st.subheader("📋 Metadata Inputs")
area = st.number_input("Area (ha)", value=1.0, format="%.2f")
sow_mon = st.number_input("Sowing Month (numeric/encoded)", value=6.0)
har_mon = st.number_input("Harvest Month (numeric/encoded)", value=12.0)
sow_to_trans_days = st.number_input("Sowing to Transplanting Days", value=25.0)
trans_to_har_days = st.number_input("Transplanting to Harvest Days", value=100.0)
expected_yield_per_ha = st.number_input(
    "Expected yield (kg/ha) under good conditions for this crop",
    value=4500.0, format="%.1f",
    help="Typical benchmark for paddy can range from 3000–6000 kg/ha. Adjust to your local expectation."
)

investment_cost_per_ha = st.number_input(
    "Expected investment cost per hectare (₹)",
    value=45000.0, format="%.1f",
    help="Include seeds, fertilizer, labour, irrigation, etc."
)


# ======================================
# --- Model Loading ---
# ======================================
cnn_model, lgbm_model = None, None

# # Handle CNN model upload safely
# if cnn_model_file is not None:
#     with tempfile.NamedTemporaryFile(delete=False, suffix=".h5") as tmp:
#         tmp.write(cnn_model_file.read())
#         tmp_path = tmp.name
#     cnn_model = tf.keras.models.load_model(tmp_path)
#     st.sidebar.success("✅ CNN model loaded successfully")
# Handle CNN model upload safely
if cnn_model_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".h5") as tmp:
        tmp.write(cnn_model_file.read())
        tmp_path = tmp.name
    # Load without compiling (fixes ValueError for unknown loss/layer)
    cnn_model = tf.keras.models.load_model(tmp_path, compile=False)
    st.sidebar.success("✅ CNN model loaded successfully")


# Handle LightGBM model upload safely
if lgbm_model_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as tmp:
        tmp.write(lgbm_model_file.read())
        tmp_path = tmp.name
    lgbm_model = joblib.load(tmp_path)
    st.sidebar.success("✅ LightGBM model loaded successfully")

# ======================================
# --- Run Prediction ---
# ======================================
if st.button("🔍 Run Prediction"):
    if not (landsat_file and sentinel_file and cnn_model and lgbm_model):
        st.error("❌ Please upload both images and both models before running prediction.")
        st.stop()

    # --- Load and preprocess Landsat ---
    with np.load(landsat_file) as ldata:
        landsat_img, ndvi_val = preprocess_landsat_image(ldata)

    # # --- CNN Feature Extraction ---
    # img_input = np.expand_dims(landsat_img, axis=0)  # shape (1, 12, 12, 5)
    # cnn_features = cnn_model.predict(img_input, verbose=0)
    # cnn_features = cnn_features.flatten()  # 1D feature vector

    # --- CNN Feature Extraction using layer 21 (same as Kaggle setup) ---
    img_input = np.expand_dims(landsat_img, axis=0)  # (1, 12, 12, 5)
    
    # Build feature extractor to match Kaggle setup
    feature_extractor = tf.keras.Model(
        inputs=cnn_model.input,
        outputs=cnn_model.layers[21].output
    )
    
    cnn_features = feature_extractor.predict(img_input, verbose=0)
    cnn_features = cnn_features.flatten()  # should be length 128
    st.write("📏 Extracted CNN feature vector shape:", cnn_features.shape)


    # --- Sentinel feature extraction ---
    with np.load(sentinel_file) as sdata:
        VV_mean, VH_mean, VH_VV_ratio, VH_VV_ratio_trans2 = compute_sentinel_features(sdata)

    # --- Transformations ---
    sow_to_trans_log = np.log1p(sow_to_trans_days).astype(np.float64)

    # --- Prepare LightGBM input ---
    tabular_features = np.array([
        area, sow_mon, har_mon, sow_to_trans_log, trans_to_har_days,
        VV_mean, VH_mean, VH_VV_ratio, VH_VV_ratio_trans2, ndvi_val
    ])
    full_input = np.concatenate([tabular_features, cnn_features])
    full_input = full_input.reshape(1, -1)

    # # --- Predict yield ---
    # yield_pred_log = float(lgbm_model.predict(full_input)[0])
    # yield_pred = np.expm1(yield_pred_log)  # inverse log1p

    # --- Safety check before prediction ---
    if hasattr(lgbm_model, 'n_features_in_') and full_input.shape[1] != lgbm_model.n_features_in_:
        st.error(f"❌ Feature size mismatch: LightGBM model expects {lgbm_model.n_features_in_} features but received {full_input.shape[1]}.")
        st.info("Please ensure your CNN feature extractor is the same one used during training.")
        st.stop()
    
    # --- Debug info (optional but helpful) ---
    st.write("📏 LightGBM model expects features:", getattr(lgbm_model, 'n_features_in_', 'unknown'))
    st.write("📏 App is sending features:", full_input.shape[1])
    st.write("📏 CNN feature vector length:", cnn_features.shape[0])
    
    # --- Predict yield ---
    yield_pred_log = float(lgbm_model.predict(full_input)[0])
    yield_pred = np.expm1(yield_pred_log)


    # --- Display results ---
    st.success(f"**Predicted Yield:** {yield_pred:.2f} kg/ha 🌾")

    st.write("### 📊 Feature Summary")
    st.dataframe({
        "AREA": [area],
        "sow_mon": [sow_mon],
        "har_mon": [har_mon],
        "sow_to_trans_days (log1p)": [sow_to_trans_log],
        "trans_to_har_days": [trans_to_har_days],
        "VV_mean": [VV_mean],
        "VH_mean": [VH_mean],
        "VH_VV_ratio": [VH_VV_ratio],
        "VH_VV_ratio_trans2": [VH_VV_ratio_trans2],
        "NDVI": [ndvi_val],
        "CNN_features_dim": [cnn_features.shape[0]]
    })

    st.balloons()

# ======================================
# --- Interpretability & Explanation ---
# ======================================
if "yield_pred" in locals():
    
    # Typical paddy price range
    import datetime
    import google.generativeai as genai
    import re
    
    # --- Step 1: Get current year using Python ---
    # current_year = datetime.datetime.now().year
    
    # --- Step 2: Configure Gemini (secure API key or fallback to hardcoded for now) ---
    # --- Secure Gemini API Key setup ---
    # genai.configure(api_key="AIzaSyBBKgwflgq7lEWn130W8BE_Qask6SYHHVo")

    # # ✅ Use a supported model
    # try:
    #     model_explainer = genai.GenerativeModel("gemini-2.0-flash")
    # except Exception:
    #     st.warning("⚠️ Unable to initialize Gemini model. Falling back to gemini-1.5-pro.")
    #     try:
    #         model_explainer = genai.GenerativeModel("gemini-1.5-pro")
    #     except:
    #         model_explainer = None
    
    # # --- Step 4: Prepare pricing dataset prompt ---
    # prompt_price = f"""
    # Based on the following Tamil Nadu paddy procurement data, return the effective procurement price
    # (including state incentive) for the {current_year}-{current_year+1} Kharif Marketing Season
    # for **Common Paddy**, expressed as a single numeric value with the unit ₹/kg and nothing else.
    # """
    
    # # --- Step 5: Query Gemini (Simplified for direct value response) ---
    # # --- Step 5: Query Gemini (Simplified for direct value response) ---
    # try:
    #     with st.spinner("📊 Fetching Tamil Nadu paddy price for the current season..."):
    #         price_response = model_price.generate_content(prompt_price)
    #         response_text = price_response.text.strip()
    
    #         # The response is usually like "₹25.00" — so clean it up
    #         response_text = (
    #             response_text.replace("₹", "")
    #                          .replace("Rs", "")
    #                          .replace("per kg", "")
    #                          .replace("/kg", "")
    #                          .strip()
    #         )
    
    #         # Parse numeric safely
    #         try:
    #             paddy_price_avg = float(response_text)
    #         except ValueError:
    #             paddy_price_avg = 25.0  # fallback if Gemini returns anything unexpected
    
    #     st.success(f"🌾 Using {current_year}-{current_year+1} KMS price: **₹{paddy_price_avg:.2f}/kg**")
    
    # except Exception as e:
    #     st.warning("⚠️ Could not fetch price from Gemini — using default ₹25.00/kg.")
    #     st.caption(str(e))
    #     paddy_price_avg = 25.0
    paddy_price_avg = 25.0

    # # --- 1️⃣ Yield Interpretation ---
    # if yield_pred < 2500:
    #     yield_text = "below average yield. This may indicate suboptimal crop health, limited soil moisture, or stress during the growing season."
    # elif 2500 <= yield_pred < 5000:
    #     yield_text = "moderate yield, typical for average crop conditions. The crop appears healthy but may not have reached full potential."
    # else:
    #     yield_text = "high yield potential. Conditions appear favorable, with strong vegetation signals and consistent canopy growth."
    
    # st.markdown(f"**Yield Assessment:** The predicted yield of `{yield_pred:.2f} kg/ha` suggests {yield_text}")

    # --- Economic analysis (scaled to total land area) ---
    # --- Economic analysis (scaled to total land area) ---
    # Convert predicted yield from kg/ha to total yield
    predicted_yield_total_kg = yield_pred * area
    
    # Compute total revenue (₹)
    predicted_revenue_total_rs = predicted_yield_total_kg * paddy_price_avg
    
    # Compute total investment (₹)
    total_investment_rs = investment_cost_per_ha * area
    
    # Profit or loss for total area
    profit_or_loss_rs = predicted_revenue_total_rs - total_investment_rs
    profit_margin_pct = (profit_or_loss_rs / total_investment_rs * 100) if total_investment_rs > 0 else 0
    
    # Yield relative to expected (for rating purposes only)
    yield_pct_of_expected = (yield_pred / expected_yield_per_ha * 100) if expected_yield_per_ha > 0 else 0
    
    # --- 📊 Yield and Economic Summary ---
    # --- 📊 Yield and Economic Summary ---
    st.subheader("📈 Yield and Economic Analysis")
    
    # Helper function to highlight text
    def highlight(text, color):
        return f"<span style='background-color:{color}; padding:2px 6px; border-radius:4px;'>{text}</span>"
    
    # Prepare highlighted key values
    yield_pct_html = highlight(f"{yield_pct_of_expected:.1f}%", "#FFD54F")  # soft yellow
    profit_pct_html = highlight(f"{profit_margin_pct:.1f}%", "#AED581")     # light green
    price_html = highlight(f"₹{paddy_price_avg:.2f}/kg", "#81C784")         # medium green
    
    # Display main info
    st.markdown(f"""
    **Predicted yield:** {yield_pred:.2f} kg/ha ({yield_pct_html} of expected)  
    **Total area:** {area:.2f} ha  
    **Predicted total yield:** {predicted_yield_total_kg:,.1f} kg  
    **Market price used:** {price_html}  
    **Predicted total revenue:** ₹{predicted_revenue_total_rs:,.0f}  
    **Total investment cost:** ₹{total_investment_rs:,.0f}  
    """, unsafe_allow_html=True)
    
    # --- Profitability insights ---
    if profit_margin_pct < 0:
        st.markdown(
            f"❌ **Loss:** ₹{abs(profit_or_loss_rs):,.0f} "
            f"(<b style='color:#E53935'>{profit_pct_html}</b> below break-even)",
            unsafe_allow_html=True,
        )
    elif profit_margin_pct < 20:
        st.markdown(
            f"⚠️ **Low Profit:** ₹{profit_or_loss_rs:,.0f} "
            f"(<b style='color:#FB8C00'>{profit_pct_html}</b> margin)",
            unsafe_allow_html=True,
        )
    elif profit_margin_pct < 50:
        st.markdown(
            f"ℹ️ **Moderate Profit:** ₹{profit_or_loss_rs:,.0f} "
            f"(<b style='color:#FDD835'>{profit_pct_html}</b> margin)",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"✅ **High Profit:** ₹{profit_or_loss_rs:,.0f} "
            f"(<b style='color:#43A047'>{profit_pct_html}</b> margin)",
            unsafe_allow_html=True,
        )
    
    # --- Yield Rating ---
    if yield_pct_of_expected < 50:
        yield_text = "Poor — yield far below potential; likely stress or resource limitation."
    elif yield_pct_of_expected < 80:
        yield_text = "Below average — moderate stress or management gaps."
    elif yield_pct_of_expected <= 110:
        yield_text = "Good — near expected performance."
    else:
        yield_text = "Excellent — favorable conditions and efficient management."
    
    st.markdown(
        f"**Yield Assessment:** {yield_text} "
        f"(Predicted: <b style='color:#64B5F6'>{yield_pred:.2f} kg/ha</b>)",
        unsafe_allow_html=True
    )
    
    # --- Economic Assessment Summary ---
    if profit_margin_pct < 0:
        st.markdown(
            f"**Economic Assessment:** "
            f"Loss of ₹{abs(profit_or_loss_rs):,.0f} "
            f"(<b style='color:#E53935'>{profit_margin_pct:.1f}%</b>) on total area.",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"**Economic Assessment:** "
            f"Profit of ₹{profit_or_loss_rs:,.0f} "
            f"(<b style='color:#43A047'>{profit_margin_pct:.1f}%</b>) on total area.",
            unsafe_allow_html=True,
        )


    
    st.subheader("🧠 Model Interpretation & Insights")
    # --- 2️⃣ NDVI Analysis ---
    if ndvi_val < 0.3:
        ndvi_text = "indicates sparse or stressed vegetation — possibly due to poor germination, drought, or nutrient stress."
    elif 0.3 <= ndvi_val < 0.6:
        ndvi_text = "represents moderate vegetation density, typical of crops in mid-growth or under mild stress."
    else:
        ndvi_text = "shows dense vegetation, healthy chlorophyll activity, and optimal photosynthetic performance."
    
    st.markdown(f"**NDVI Insight:** NDVI = `{ndvi_val:.3f}` → {ndvi_text}")
    
    # --- 3️⃣ Radar Reflectance Analysis (Sentinel VV/VH) ---
    if VH_VV_ratio < 0.4:
        radar_text = "suggests a well-developed canopy with minimal soil exposure, indicating good vegetation cover."
    elif 0.4 <= VH_VV_ratio < 0.8:
        radar_text = "indicates moderate backscatter, consistent with balanced crop density and moisture."
    else:
        radar_text = "shows high backscatter, which could mean surface roughness, high moisture, or sparse vegetation."
    
    st.markdown(f"**Radar Backscatter Insight:** VH/VV ratio = `{VH_VV_ratio:.3f}` → {radar_text}")
    
    # --- 4️⃣ Temporal Metadata Analysis ---
    month_names = {
        1: "January", 2: "February", 3: "March", 4: "April", 5: "May", 6: "June",
        7: "July", 8: "August", 9: "September", 10: "October", 11: "November", 12: "December"
    }
    sow_m = month_names.get(int(sow_mon), f"Month {sow_mon}")
    har_m = month_names.get(int(har_mon), f"Month {har_mon}")
    st.markdown(f"**Temporal Insight:** Crop sown in **{sow_m}** and harvested around **{har_m}**. "
                f"Total growth duration: ~`{sow_to_trans_days + trans_to_har_days}` days, typical for seasonal crops like paddy or maize.")
    
    # --- 5️⃣ Overall Summary ---
    st.info(f"""
    **Summary Interpretation**
    - Predicted yield: `{yield_pred:.2f} kg/ha` → {yield_text}
    - NDVI: `{ndvi_val:.3f}` → {ndvi_text}
    - Radar VH/VV ratio: `{VH_VV_ratio:.3f}` → {radar_text}
    - Sowing–harvest period: {sow_m} to {har_m} (~{sow_to_trans_days + trans_to_har_days} days)
    """)

# ======================================
# --- AI Explanation (via Gemini API) ---
# ======================================
if "yield_pred" in locals() and "ndvi_val" in locals():
    import google.generativeai as genai
    import os

    # --- Secure Gemini API Key setup ---
    genai.configure(api_key="AIzaSyBBKgwflgq7lEWn130W8BE_Qask6SYHHVo")

    # ✅ Use a supported model
    try:
        model_explainer = genai.GenerativeModel("gemini-2.0-flash")
    except Exception:
        st.warning("⚠️ Unable to initialize Gemini model. Falling back to gemini-1.5-pro.")
        try:
            model_explainer = genai.GenerativeModel("gemini-1.5-pro")
        except:
            model_explainer = None

    # --- Prepare context-rich AI prompt ---
    ai_prompt = f"""
    You are an expert agronomist and data scientist.
    Based on the following crop and satellite analysis results, explain the findings
    and provide clear, actionable recommendations to the farmer.

    ---
    **Input Data Summary:**
    - Area: {area:.2f} ha
    - Sowing Month: {sow_mon}
    - Harvest Month: {har_mon}
    - Sowing → Transplant Days: {sow_to_trans_days}
    - Transplant → Harvest Days: {trans_to_har_days}

    **Computed Satellite Metrics:**
    - NDVI: {ndvi_val:.3f}
    - VV_mean: {VV_mean:.3f}
    - VH_mean: {VH_mean:.3f}
    - VH/VV ratio: {VH_VV_ratio:.3f}
    - Power-transformed ratio: {VH_VV_ratio_trans2:.3f}

    **Predicted Yield:**
    - {yield_pred:.2f} kg/ha

    ---
    Generate a clear, well-structured explanation that includes:
    1️⃣ A short friendly greeting.  
    2️⃣ NDVI-based interpretation (crop vigor and greenness).  
    3️⃣ Radar-based interpretation (moisture, canopy structure, surface texture).  
    4️⃣ Agronomic recommendations (irrigation, nutrient, timing).  
    5️⃣ Final yield assessment and motivational note.  
    Avoid technical jargon and explain in a farmer-friendly manner.
    """

    if model_explainer:
        try:
            with st.spinner("🧠 Generating expert interpretation using Gemini..."):
                ai_response = model_explainer.generate_content(ai_prompt)
            st.subheader("🌿 AI-Powered Agronomic Advisory")
            st.write(ai_response.text)
        except Exception as e:
            st.warning("⚠️ AI advisory unavailable. The Gemini model could not generate a response.")
            st.caption(str(e))
    else:
        st.info("💡 AI explanation skipped — no compatible Gemini model was initialized.")

else:
    st.info("👆 Run the prediction first to generate the AI advisory.")

