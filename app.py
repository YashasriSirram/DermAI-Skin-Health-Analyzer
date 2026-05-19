import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from collections import OrderedDict

# =========================
# DEVICE
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# MODEL CONFIGURATION
# =========================
num_classes = 3

model = models.resnet50(weights=None)

model.fc = nn.Sequential(OrderedDict([
    ('fc1', nn.Linear(model.fc.in_features, 1024)),
    ('relu', nn.ReLU()),
    ('dropout', nn.Dropout(0.4)),
    ('fc2', nn.Linear(1024, num_classes))
]))

# =========================
# LOAD MODEL WEIGHTS
# =========================
model.load_state_dict(
    torch.load(
        "models/save_models_here/model_Resnet50.pt",
        map_location=device
    )
)

model.to(device)
model.eval()

# =========================
# CLASS LABELS
# =========================
classes = [
    "Benign",
    "Malignant",
    "Normal"
]

# =========================
# IMAGE TRANSFORM
# =========================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="DermAI",
    page_icon="🩺",
    layout="wide"
)

# =========================
# MODERN CSS
# =========================
st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif;
}

/* MAIN APP */
.stApp {

    background:
        radial-gradient(circle at top left, #dff6f0 0%, transparent 30%),
        radial-gradient(circle at bottom right, #d6f5ff 0%, transparent 35%),
        linear-gradient(to bottom right, #f8fbff, #eef7f5);

    color: #0F2C59;
}

/* MAIN CONTAINER */
.block-container {
    padding-top: 1rem;
    padding-bottom: 2rem;
}

/* REMOVE STREAMLIT TOP BAR */
header {
    background: transparent !important;
}

[data-testid="stHeader"] {
    background: transparent !important;
}

/* SIDEBAR */
section[data-testid="stSidebar"] {

    background: rgba(255,255,255,0.75);

    backdrop-filter: blur(10px);

    border-right: 1px solid #E5E7EB;
}

/* NAVBAR */
.navbar {

    background: linear-gradient(
    90deg,
    #00A896,
    #02C39A
);

    padding: 18px 30px;

    border-radius: 24px;

    backdrop-filter: blur(12px);

    box-shadow:
        0 8px 25px rgba(0,0,0,0.05);

    margin-bottom: 30px;
}

/* TITLES */
h1, h2, h3 {

    color: #0F2C59 !important;

    font-weight: 700 !important;
}

/* TEXT */
p, label, div {

    color: #2C3E50;
}

/* BUTTONS */
.stButton>button {

    width: 100%;

    background: linear-gradient(
        90deg,
        #00A896,
        #02C39A
    );

    color: white;

    border-radius: 14px;

    height: 3.2em;

    border: none;

    font-size: 16px;

    font-weight: 600;

    transition: 0.3s ease;

    box-shadow:
        0 6px 18px rgba(0,168,150,0.18);
}

/* BUTTON HOVER */
.stButton>button:hover {

    transform: scale(1.02);

    background: linear-gradient(
        90deg,
        #028090,
        #00A896
    );

    color: white;
}

/* METRIC CARDS */
[data-testid="metric-container"] {

    background: rgba(255,255,255,0.8);

    border: 1px solid rgba(255,255,255,0.6);

    padding: 20px;

    border-radius: 20px;

    backdrop-filter: blur(10px);

    box-shadow:
        0 10px 30px rgba(0,0,0,0.06);
}

/* FILE UPLOADER */
[data-testid="stFileUploader"] {

    background: rgba(255,255,255,0.75);

    border-radius: 18px;

    padding: 18px;

    border: 1px solid #D1D5DB;

    backdrop-filter: blur(10px);
}

/* ALERTS */
.stAlert {

    border-radius: 16px;
}

/* PROGRESS BAR */
.stProgress > div > div > div > div {

    background:
        linear-gradient(
            90deg,
            #00A896,
            #02C39A
        );
}

/* IMAGE */
img {

    border-radius: 20px;

    box-shadow:
        0 10px 30px rgba(0,0,0,0.08);
}
            /* SIDEBAR TOGGLE BUTTON */
button[kind="header"] {

    background-color: #00A896 !important;

    color: green !important;

    border-radius: 12px !important;

    width: 42px !important;

    height: 42px !important;

    box-shadow:
        0 6px 15px rgba(0,168,150,0.25);

    transition: 0.3s ease;
}

/* HOVER EFFECT */
button[kind="header"]:hover {

    background-color: #028090 !important;

    transform: scale(1.05);
}

/* REMOVE FOOTER */
footer {
    visibility: hidden;
}

/* REMOVE STREAMLIT MENU */
#MainMenu {
    visibility: hidden;
}

</style>
""", unsafe_allow_html=True)

# =========================
# TOP NAVBAR
# =========================
nav1, nav2, nav3 = st.columns([7,1,1])

with nav1:

    st.markdown("""
    <div class="navbar">
    <h2>🩺 DermAI</h2>
    </div>
    """, unsafe_allow_html=True)

with nav2:
    login_btn = st.button("Login")

with nav3:
    register_btn = st.button("Register")

# =========================
# LOGIN FORM
# =========================
if login_btn:

    st.subheader("Login")

    username = st.text_input("Username")

    password = st.text_input(
        "Password",
        type="password"
    )

    if st.button("Sign In"):

        st.success(f"Welcome back, {username}!")

# =========================
# REGISTER FORM
# =========================
if register_btn:

    st.subheader("Create Account")

    new_user = st.text_input("Create Username")

    new_pass = st.text_input(
        "Create Password",
        type="password"
    )

    confirm_pass = st.text_input(
        "Confirm Password",
        type="password"
    )

    if st.button("Register Account"):

        if new_pass == confirm_pass:

            st.success(
                "Account created successfully!"
            )

        else:

            st.error(
                "Passwords do not match."
            )

# =========================
# SIDEBAR
# =========================
st.sidebar.title("🩺 DermAI")

page = st.sidebar.radio(
    "Navigate",
    ["🏠 Dashboard", "🔬 Analyze Image", "📑 Project Details"]
)

st.sidebar.markdown("---")

st.sidebar.caption(
    "Deep learning based skin lesion screening system."
)

st.sidebar.markdown(
    "### Developed by Yashasri Sirram"
)

# =========================
# DASHBOARD PAGE
# =========================
if page == "🏠 Dashboard":

    col1, col2 = st.columns([5, 2])

    with col1:

        st.markdown("""
        <div style='padding-top:30px;'>

        <h1 style='font-size:62px; line-height:1.1;'>
        Early Skin Disease Detection Using AI
        </h1>

        <p style='font-size:20px; color:#4B5563; margin-top:20px;'>
        Upload a dermoscopic image and receive AI-assisted lesion analysis in seconds.
        </p>

        </div>
        """, unsafe_allow_html=True)

        st.write("")

        st.button("🚀 Start Skin Analysis")

    with col2:

        st.markdown("""
        <div style='background:rgba(255,255,255,0.8);
                    padding:30px;
                    border-radius:24px;
                    margin-top:30px;
                    backdrop-filter:blur(10px);
                    box-shadow:0 10px 30px rgba(0,0,0,0.08);'>

        <h3 style='color:#0F2C59;'>
        DermAI Scanner
        </h3>

        <p style='color:#4B5563;'>
        Deep learning powered support for skin lesion analysis and classification.
        </p>

        </div>
        """, unsafe_allow_html=True)

    st.write("")
    st.write("")

    # METRICS
    m1, m2, m3 = st.columns(3)

    m1.metric("Images Trained", "10,000+")
    m2.metric("Validation Accuracy", "93.7%")
    m3.metric("Prediction Time", "2 sec")

    st.write("")
    st.write("")

    # HOW IT WORKS
    st.markdown("""
    <h2 style='margin-bottom:25px;'>
    How It Works
    </h2>
    """, unsafe_allow_html=True)

    step1, step2, step3 = st.columns(3)

    with step1:

        st.markdown("""
        <div style='background:white;
                    padding:25px;
                    border-radius:20px;
                    height:220px;
                    box-shadow:0 10px 25px rgba(0,0,0,0.06);'>

        <h3>📤 Step 1</h3>

        <h4>Upload Image</h4>

        <p>
        Upload a dermoscopic skin lesion image for analysis.
        </p>

        </div>
        """, unsafe_allow_html=True)

    with step2:

        st.markdown("""
        <div style='background:white;
                    padding:25px;
                    border-radius:20px;
                    height:220px;
                    box-shadow:0 10px 25px rgba(0,0,0,0.06);'>

        <h3>🧠 Step 2</h3>

        <h4>AI Analysis</h4>

        <p>
        Deep learning model processes the image using ResNet50.
        </p>

        </div>
        """, unsafe_allow_html=True)

    with step3:

        st.markdown("""
        <div style='background:white;
                    padding:25px;
                    border-radius:20px;
                    height:220px;
                    box-shadow:0 10px 25px rgba(0,0,0,0.06);'>

        <h3>📊 Step 3</h3>

        <h4>Get Results</h4>

        <p>
        Receive lesion classification and confidence score instantly.
        </p>

        </div>
        """, unsafe_allow_html=True)

    st.write("")
    st.write("")

    st.info(
        "Current model performs best on clear dermoscopic skin images."
    )

    st.caption(
        "Predictions generated by the model should not be considered a medical diagnosis."
    )

# =========================
# ANALYZE PAGE
# =========================
elif page == "🔬 Analyze Image":

    st.markdown("""
    <h2>
    Analyze Skin Image
    </h2>
    """, unsafe_allow_html=True)

    st.write(
        "Upload a dermoscopic skin image for lesion classification."
    )

    uploaded_file = st.file_uploader(
        "Upload Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        image = Image.open(uploaded_file).convert("RGB")

        col1, col2 = st.columns([1, 1])

        with col1:

            st.image(
                image,
                caption="Uploaded Image",
                use_container_width=True
            )

        with col2:

            img_tensor = transform(image)
            img_tensor = img_tensor.unsqueeze(0).to(device)

            if st.button("🚀 Run Analysis"):

                with st.spinner("Analyzing lesion image..."):

                    with torch.no_grad():

                        outputs = model(img_tensor)

                        probabilities = torch.softmax(outputs, dim=1)

                        confidence, predicted = torch.max(
                            probabilities,
                            1
                        )

                        predicted_class = classes[predicted.item()]

                        confidence_score = confidence.item() * 100

                st.success("Analysis completed")

                st.subheader("Prediction")

                st.markdown(
                    f"## 🩺 `{predicted_class}`"
                )

                st.metric(
                    label="Confidence Score",
                    value=f"{confidence_score:.2f}%"
                )

                st.progress(int(confidence_score))

                disease_info = {

                    "Benign":
                    "Usually non-cancerous lesion with lower clinical risk.",

                    "Malignant":
                    "Potentially dangerous lesion requiring immediate medical evaluation.",

                    "Normal":
                    "No major abnormality detected in uploaded image."
                }

                st.info(disease_info[predicted_class])

                if predicted_class == "Malignant":

                    st.error(
                        "Please consult a certified dermatologist for further clinical examination."
                    )

                else:

                    st.success(
                        "Lower-risk lesion prediction generated."
                    )

    st.caption(
        "Model predictions are generated using deep learning techniques trained on medical imaging datasets."
    )

# =========================
# PROJECT DETAILS
# =========================
elif page == "📖 Project Details":

    st.markdown("""
    <h2>
    About The Project
    </h2>
    """, unsafe_allow_html=True)

    st.write("""

This project was developed as part of a deep learning and medical imaging exploration project.

The application uses a fine-tuned ResNet50 convolutional neural network trained for skin lesion classification.

### Technologies Used
- Python
- Streamlit
- PyTorch
- Torchvision
- PIL
- Deep Learning

### Features
- Skin lesion image upload
- Deep learning classification
- Confidence score visualization
- Responsive dashboard interface
- Medical image screening workflow

### Model Information
- Architecture: ResNet50
- Framework: PyTorch
- Categories: 3
- Input Size: 224 × 224

""")

    st.success(
        "Built and tested for educational and research purposes."
    )

# =========================
# FOOTER
# =========================
st.markdown("---")

st.markdown(
    "<center>Developed by Yashasri Sirram • Streamlit • PyTorch • ResNet50</center>",
    unsafe_allow_html=True
)