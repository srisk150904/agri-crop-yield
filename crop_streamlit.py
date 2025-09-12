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

import streamlit as st
import numpy as np

# --- Dummy model for frontend testing ---
class DummyModel:
    def predict(self, inputs):
        img_array, meta_array = inputs
        # Fake classification: always "Paddy" with 70% confidence
        class_pred = np.array([[0.3, 0.7]])
        # Fake regression: yield = area * 100 (mock formula)
        reg_pred = np.array([[meta_array[0][0] * 100]])
        return class_pred, reg_pred

# Swap with your real model later
# model = tf.keras.models.load_model("cropnet_model.h5")
model = DummyModel()

# --- Streamlit UI ---
st.title("🌾 Crop Classification & Yield Prediction")
st.write("Upload a Landsat patch (.npz) with 5 bands and enter metadata to classify crop type and predict yield.")

# --- File uploader ---
uploaded_file = st.file_uploader("Upload Landsat Patch (.npz)", type=["npz"])

# --- Metadata input ---
st.subheader("Metadata Inputs")

area = st.number_input("Area (ha)", value=1.0, format="%.2f", key="area_input")
sowing_month = st.number_input("Sowing Month (numeric/encoded)", value=100.0, key="sowing_month_input")
harvest_month = st.number_input("Harvest Month (numeric/encoded)", value=200.0, key="harvest_month_input")
sowing_to_transplanting_days = st.number_input(
    "Sowing to Transplanting Days (numeric/encoded)", value=200.0, key="sowing_to_transplanting_input"
)

if uploaded_file is not None:
    with np.load(uploaded_file) as data:
        # Load first array (usually stored as 'arr_0')
        img_array = data[list(data.keys())[0]]

    st.write("✅ Landsat patch loaded:", img_array.shape)

    # Normalize (Landsat reflectance often 0–10000)
    img_array = img_array.astype("float32")
    if img_array.max() > 1.0:
        img_array = img_array / 10000.0  

    # Ensure shape = (H, W, C)
    if img_array.ndim == 2:
        img_array = np.expand_dims(img_array, axis=-1)
    elif img_array.shape[0] < img_array.shape[-1]:
        img_array = np.transpose(img_array, (1, 2, 0))

    # --- Pseudo-RGB Preview (using SR_B4, SR_B5, SR_B6) ---
    if img_array.shape[-1] >= 3:
        pseudo_rgb = np.stack([
            img_array[:, :, 0],  # SR_B4 (Red)
            img_array[:, :, 1],  # SR_B5 (NIR)
            img_array[:, :, 2],  # SR_B6 (SWIR1)
        ], axis=-1)
        pseudo_rgb = (pseudo_rgb - pseudo_rgb.min()) / (pseudo_rgb.max() - pseudo_rgb.min() + 1e-6)
        st.subheader("Pseudo-RGB Preview (SR_B4, SR_B5, SR_B6)")
        st.image(pseudo_rgb, caption="Landsat Patch (Pseudo-RGB)", use_column_width=True)

    # Expand dims for model
    img_array_batch = np.expand_dims(img_array, axis=0)   # (1, H, W, 5)
    meta_array = np.array([[area, sowing, harvest]])

    # --- Prediction button ---
    if st.button("🔍 Run Prediction"):
        class_pred, reg_pred = model.predict([img_array_batch, meta_array])
        class_label = int(np.argmax(class_pred, axis=1)[0])
        yield_pred = float(reg_pred[0][0])

        # Map labels (example: 0=Non-paddy, 1=Paddy)
        label_map = {0: "Non-Paddy", 1: "Paddy"}

        st.success(f"**Crop Type:** {label_map[class_label]}")
        st.info(f"**Predicted Yield:** {yield_pred:.2f} kg/ha")
