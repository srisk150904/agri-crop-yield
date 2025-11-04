import streamlit as st
import os #picking up env vars
from PIL import Image
import google.generativeai as genai

genai.configure(api_key="AIzaSyBBKgwflgq7lEWn130W8BE_Qask6SYHHVo")


#function to load gemini pro vision
model=genai.GenerativeModel('gemini-1.5-flash')


def get_gemini_response(Input,Image,prompt):
    response=model.generate_content((Input,Image[0],prompt))
    return response.text


def input_image_details(uploaded_file):
    if uploaded_file is not None:
        bytes_data=uploaded_file.getvalue()
        image_parts=[
            {
                "mime_type":uploaded_file.type,
                "data":bytes_data
            }
        ]
        return image_parts
    else:
        raise FileNotFoundError("no file uploaded")


#init streamlit
st.set_page_config(page_title="WIE's invoice generator")
st.sidebar.header("RoboBill🦾")
st.sidebar.write("Made by <your name>.")
st.sidebar.write("Assistant used is Gemini Pro Vision.")
st.header("RoboBill 🦾")
st.subheader("Manage your expenses with the help of the robot!")
input=st.text_input("What do you want me to do? ",key="input")
uploaded_file = st.file_uploader("Choose an image.", type=["jpg","jpeg","png"])
image=""
if uploaded_file is not None:
    image=Image.open(uploaded_file)
    st.image(image,caption="Uploaded Image", use_column_width=True)


ssubmit=st.button("Let's go!")
Input_prompt="""
You are an expert in understanding invoices. 
We will upload a image as invoices 
and you will have to answer any questions based on the uploaded invoice image
Make sure to greet the user first and then provide the information as suited.
Make sure to keep the font uniform and give the items list in a point-wise format.
At the end, make sure to repeat the name of our app "RoboBill 🦾" and ask the user to use it again.
"""


#button prog
if ssubmit:
    image_data=input_image_details(uploaded_file)
    response=get_gemini_response(Input_prompt,image_data,input)
    st.subheader("Here's what you need to know:")
    st.write(response)


# ======================================
# --- AI Explanation (via Gemini API) ---
# ======================================
import google.generativeai as genai
import os

# 🔐 Set up Gemini API key securely from environment variable
genai.configure(api_key=st.secrets.get("GEMINI_API_KEY", None) or os.getenv("GEMINI_API_KEY"))

try:
    model_explainer = genai.GenerativeModel("gemini-1.5-flash")

    # Prepare AI prompt with context
    ai_prompt = f"""
    You are an expert agronomist and data scientist.
    Based on the following crop and satellite analysis results, explain the findings
    and provide recommendations to the farmer in clear and actionable terms.

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
    Now generate a well-structured explanation that includes:
    1. A friendly greeting and brief summary of the crop health.
    2. NDVI-based interpretation (vegetation vigor and greenness).
    3. Radar reflectance interpretation (moisture, canopy density).
    4. Possible agronomic insights (irrigation, nutrient, or timing suggestions).
    5. Overall yield assessment and actionable recommendations.
    6. A short motivational closing note for the farmer.
    """

    # Call Gemini API
    with st.spinner("🧠 Generating expert interpretation using Gemini..."):
        ai_response = model_explainer.generate_content(ai_prompt)

    st.subheader("🌿 AI-Powered Agronomic Advisory")
    st.write(ai_response.text)

except Exception as e:
    st.warning("⚠️ AI advisory unavailable. Please check your Gemini API key or network connection.")
    st.write(e)

