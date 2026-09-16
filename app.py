import streamlit as st
import numpy as np
import cv2
from PIL import Image
import pandas as pd
import plotly.express as px
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Secure AI Defense System",
    page_icon="🛡️",
    layout="wide"
)

# ============================================================
# SESSION STATE
# ============================================================
if "history" not in st.session_state:
    st.session_state.history = []

if "total_scans" not in st.session_state:
    st.session_state.total_scans = 0

if "threats" not in st.session_state:
    st.session_state.threats = 0

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# ============================================================
# CUSTOM CSS
# ============================================================
st.markdown(
    """
    <style>
    .main-title {
        background: linear-gradient(135deg, #00c6ff, #0072ff);
        padding: 25px;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 30px;
    }

    .main-title h1 {
        color: white;
        margin: 0;
        font-size: 2.5rem;
    }

    .main-title p {
        color: rgba(255,255,255,0.9);
        margin: 10px 0 0 0;
    }

    .metric-card {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border-radius: 15px;
        padding: 15px;
        text-align: center;
        border: 1px solid rgba(0,198,255,0.3);
    }

    .info-box {
        padding: 15px;
        border-radius: 12px;
        border: 1px solid rgba(0,198,255,0.3);
        margin: 10px 0;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="main-title">
        <h1>🛡️ Secure AI Defense System</h1>
        <p>Multi-layer image and AI security analysis dashboard</p>
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("## 🎮 Navigation")

    page = st.radio(
        "Select Module",
        [
            "📊 Dashboard",
            "🖼️ AI Image Detector",
            "👤 Face Spoof Detector",
            "⚠️ Adversarial Detector",
            "💬 Secure Chatbot",
            "📜 Detection History"
        ]
    )

    st.markdown("---")
    st.markdown("## 📊 Live Stats")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Total Scans", st.session_state.total_scans)

    with col2:
        st.metric("Threats Found", st.session_state.threats)

    if st.session_state.total_scans > 0:
        safety_rate = (
            (st.session_state.total_scans - st.session_state.threats)
            / st.session_state.total_scans
        ) * 100
        st.progress(
            max(0, min(1, safety_rate / 100)),
            text=f"Safety Rate: {safety_rate:.1f}%"
        )

    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.markdown(
        """
        **Secure AI Defense System**

        B.Tech CSE Project

        **Developer:** Simran Kumari

        Modules:
        - AI Image Analysis
        - Face Spoof Analysis
        - Adversarial Image Analysis
        - Prompt Security
        - Detection History
        """
    )


# ============================================================
# HELPER FUNCTIONS
# ============================================================
def log_detection(module, result, confidence, details):
    """Store a detection result in the current session."""
    st.session_state.total_scans += 1

    if result in ["AI", "Spoof", "Attack", "Suspicious", "NoFace"]:
        st.session_state.threats += 1

    details_text = str(details)

    st.session_state.history.append(
        {
            "Time": datetime.now().strftime("%H:%M:%S"),
            "Date": datetime.now().strftime("%Y-%m-%d"),
            "Module": module,
            "Result": result,
            "Confidence": round(float(confidence), 1),
            "Details": (
                details_text[:100] + "..."
                if len(details_text) > 100
                else details_text
            ),
        }
    )


def convert_to_rgb(image):
    """Convert uploaded image safely to RGB."""
    if image.mode == "RGBA":
        rgb = Image.new("RGB", image.size, (255, 255, 255))
        rgb.paste(image, mask=image.getchannel("A"))
        return rgb

    return image.convert("RGB")


# ============================================================
# AI IMAGE DETECTION
# NOTE: This is a heuristic image-analysis module, not a
# trained AI classifier. Scores are analysis scores.
# ============================================================
def detect_ai_image(image):
    image = convert_to_rgb(image)

    # Resize very large images for faster analysis
    image.thumbnail((1200, 1200))

    img = np.array(image)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    texture = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    edges = float(np.mean(cv2.Canny(gray, 100, 200)))
    noise = float(np.var(gray))

    # FFT analysis
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude = 20 * np.log(np.abs(fshift) + 1)

    h, w = magnitude.shape

    if h > 100 and w > 100:
        high_freq = float(np.mean(magnitude[h // 4:3 * h // 4, w // 4:3 * w // 4]))
        low_freq = float(np.mean(magnitude[:h // 4, :w // 4]))
    else:
        high_freq = float(np.mean(magnitude))
        low_freq = float(np.mean(magnitude))

    freq_ratio = high_freq / (low_freq + 1e-6)

    # Color analysis
    pixels = img.reshape(-1, 3)
    unique_colors = len(np.unique(pixels, axis=0))
    color_ratio = unique_colors / max(1, img.shape[0] * img.shape[1])

    score = 0
    details = []

    if texture < 100:
        score += 25
        details.append(
            f"• Low texture variation: {texture:.0f}"
        )

    if edges < 35:
        score += 25
        details.append(
            f"• Low edge density: {edges:.1f}"
        )

    if noise < 600:
        score += 20
        details.append(
            f"• Low pixel variance: {noise:.0f}"
        )

    if freq_ratio > 1.3:
        score += 20
        details.append(
            f"• Frequency-domain anomaly: {freq_ratio:.2f}"
        )

    if color_ratio < 0.0005:
        score += 10
        details.append(
            f"• Low color diversity: {unique_colors} unique colors"
        )

    score = min(score, 100)

    if score > 60:
        result = "🤖 AI GENERATED"
        result_type = "AI"
        status = "fake"
    elif score > 35:
        result = "⚠️ SUSPICIOUS"
        result_type = "Suspicious"
        status = "suspicious"
    else:
        result = "✅ LIKELY REAL"
        result_type = "Real"
        status = "real"

    if not details:
        details.append("• No strong heuristic anomaly detected.")

    return result, score, details, result_type, status


# ============================================================
# FACE SPOOF DETECTION
# NOTE: This is a heuristic liveness/spoof analysis and should
# not be presented as production biometric authentication.
# ============================================================
def detect_face_spoof(image):
    image = convert_to_rgb(image)
    image.thumbnail((1200, 1200))

    img = np.array(image)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    face_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    eye_path = cv2.data.haarcascades + "haarcascade_eye.xml"

    face_cascade = cv2.CascadeClassifier(face_path)
    eye_cascade = cv2.CascadeClassifier(eye_path)

    if face_cascade.empty() or eye_cascade.empty():
        return (
            "❌ DETECTOR UNAVAILABLE",
            0,
            ["OpenCV Haar cascade files could not be loaded."],
            "NoFace",
            "suspicious",
        )

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60)
    )

    if len(faces) == 0:
        return (
            "❌ NO FACE DETECTED",
            0,
            ["No frontal face was detected in the uploaded image."],
            "NoFace",
            "suspicious",
        )

    spoof_score = 0
    details = []

    # Analyze the largest detected face
    x, y, w, h = max(faces, key=lambda rect: rect[2] * rect[3])

    face = gray[y:y + h, x:x + w]
    face_rgb = img[y:y + h, x:x + w]

    if face.size == 0:
        return (
            "❌ INVALID FACE REGION",
            0,
            ["The detected face region could not be analyzed."],
            "NoFace",
            "suspicious",
        )

    # Blur detection
    blur = float(cv2.Laplacian(face, cv2.CV_64F).var())

    if blur < 60:
        spoof_score += 30
        details.append(
            f"• High blur: {blur:.0f}"
        )
    elif blur < 100:
        spoof_score += 15
        details.append(
            f"• Moderate blur: {blur:.0f}"
        )

    # Edge density
    edges = cv2.Canny(face, 50, 150)
    edge_density = float(np.mean(edges > 0))

    if edge_density < 0.02:
        spoof_score += 20
        details.append(
            f"• Low edge detail: {edge_density:.3f}"
        )

    # HSV analysis
    hsv = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2HSV)
    mean_hue = float(np.mean(hsv[:, :, 0]))

    if mean_hue < 5 or mean_hue > 25:
        spoof_score += 20
        details.append(
            f"• Unusual average hue: {mean_hue:.1f}"
        )

    # Eye detection
    eyes = eye_cascade.detectMultiScale(
        face,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(15, 15)
    )

    if len(eyes) < 2:
        spoof_score += 20
        details.append(
            f"• Fewer than two eyes detected: {len(eyes)}"
        )

    # Simple frequency analysis
    f = np.fft.fft2(face)
    fshift = np.fft.fftshift(f)
    magnitude = float(np.mean(np.abs(fshift)))

    if magnitude > 80:
        spoof_score += 10
        details.append("• High-frequency pattern detected")

    spoof_score = min(spoof_score, 100)

    if spoof_score > 55:
        result = "❌ SPOOF INDICATORS DETECTED"
        result_type = "Spoof"
        status = "fake"
        confidence = spoof_score
    elif spoof_score > 30:
        result = "⚠️ SUSPICIOUS FACE"
        result_type = "Suspicious"
        status = "suspicious"
        confidence = spoof_score
    else:
        result = "✅ NO STRONG SPOOF INDICATORS"
        result_type = "Real"
        status = "real"
        confidence = 100 - spoof_score

    if not details:
        details.append("• No strong spoof indicators detected.")

    return result, confidence, details, result_type, status


# ============================================================
# ADVERSARIAL IMAGE ANALYSIS
# NOTE: This uses image statistics/heuristics. It does not
# identify a specific FGSM/PGD/CW attack with certainty.
# ============================================================
def detect_adversarial(image):
    image = convert_to_rgb(image)
    image.thumbnail((1200, 1200))

    img = np.array(image)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Noise analysis
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    noise = gray.astype(np.float32) - blurred.astype(np.float32)
    noise_std = float(np.std(noise))

    # Gradient analysis
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    grad_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)

    mean_grad = float(np.mean(grad_magnitude))
    std_grad = float(np.std(grad_magnitude))

    grad_anomaly = std_grad / (mean_grad + 1)

    # FFT high-frequency analysis
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)

    rows, cols = gray.shape
    crow, ccol = rows // 2, cols // 2

    y1 = max(0, crow - 20)
    y2 = min(rows, crow + 20)
    x1 = max(0, ccol - 20)
    x2 = min(cols, ccol + 20)

    high_freq = float(np.mean(np.abs(fshift[y1:y2, x1:x2])))

    attack_score = 0
    details = []

    if noise_std > 25:
        attack_score += 30
        details.append(
            f"• High noise level: {noise_std:.1f}"
        )
    elif noise_std > 15:
        attack_score += 15
        details.append(
            f"• Elevated noise level: {noise_std:.1f}"
        )

    if grad_anomaly > 2.5:
        attack_score += 35
        details.append(
            f"• Gradient anomaly: {grad_anomaly:.2f}"
        )

    if high_freq > 300:
        attack_score += 35
        details.append(
            f"• High-frequency energy: {high_freq:.0f}"
        )

    attack_score = min(attack_score, 100)

    if attack_score > 60:
        result = "⚠️ POSSIBLE ADVERSARIAL PERTURBATION"
        result_type = "Attack"
        status = "fake"
    elif attack_score > 35:
        result = "⚠️ SUSPICIOUS"
        result_type = "Suspicious"
        status = "suspicious"
    else:
        result = "✅ NO STRONG ATTACK INDICATORS"
        result_type = "Clean"
        status = "real"

    if not details:
        details.append("• No strong statistical anomaly detected.")

    return result, attack_score, details, result_type, status


# ============================================================
# DASHBOARD
# ============================================================
if page == "📊 Dashboard":
    st.markdown("## 📊 Live Dashboard")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            f"""
            <div class="metric-card">
                <h3>📸 Total Scans</h3>
                <h2>{st.session_state.total_scans}</h2>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            f"""
            <div class="metric-card">
                <h3>⚠️ Threats Found</h3>
                <h2>{st.session_state.threats}</h2>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:
        safety = (
            (st.session_state.total_scans - st.session_state.threats)
            / max(1, st.session_state.total_scans)
        ) * 100

        st.markdown(
            f"""
            <div class="metric-card">
                <h3>✅ Safety Rate</h3>
                <h2>{safety:.1f}%</h2>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col4:
        st.markdown(
            """
            <div class="metric-card">
                <h3>🧩 Modules</h3>
                <h2>5</h2>
            </div>
            """,
            unsafe_allow_html=True
        )

    if st.session_state.history:
        st.markdown("### 📈 Recent Activity")

        df = pd.DataFrame(st.session_state.history[-10:])

        fig = px.bar(
            df,
            x="Time",
            y="Confidence",
            color="Result",
            title="Recent Detection Scores"
        )

        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 🚀 Available Modules")

    qc1, qc2, qc3 = st.columns(3)

    with qc1:
        st.info(
            "🖼️ **AI Image Detector**\n\n"
            "Analyzes texture, edges, noise, color diversity and frequency patterns."
        )

    with qc2:
        st.info(
            "👤 **Face Spoof Detector**\n\n"
            "Checks detected faces for blur, edge, color, eye and frequency indicators."
        )

    with qc3:
        st.info(
            "⚠️ **Adversarial Detector**\n\n"
            "Analyzes noise, gradients and frequency-domain characteristics."
        )


# ============================================================
# AI IMAGE DETECTOR PAGE
# ============================================================
elif page == "🖼️ AI Image Detector":
    st.markdown("## 🖼️ AI Image Detector")
    st.caption(
        "Heuristic analysis of image characteristics. "
        "This is not a trained AI-generated-image classifier."
    )

    col1, col2 = st.columns([1, 1])

    with col1:
        uploaded_file = st.file_uploader(
            "Upload Image",
            type=["png", "jpg", "jpeg", "webp"],
            key="ai_upload"
        )

        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(
                image,
                caption="Uploaded Image",
                use_container_width=True
            )

    with col2:
        if uploaded_file:
            if st.button(
                "🔍 Analyze Image",
                type="primary",
                use_container_width=True
            ):
                with st.spinner("Analyzing image..."):
                    result, score, details, result_type, status = detect_ai_image(image)

                log_detection(
                    "AI Image Detector",
                    result_type,
                    score,
                    details
                )

                if status == "real":
                    st.success(f"### {result}")
                elif status == "fake":
                    st.error(f"### {result}")
                else:
                    st.warning(f"### {result}")

                st.metric(
                    "AI Analysis Score",
                    f"{score:.1f}%"
                )

                with st.expander(
                    "🔍 Detailed Analysis",
                    expanded=True
                ):
                    st.progress(
                        score / 100,
                        text=f"Analysis Score: {score:.1f}%"
                    )

                    st.markdown("#### 📋 Indicators")

                    for detail in details:
                        st.write(detail)

                st.info(
                    "ℹ️ A high score indicates that this heuristic analysis "
                    "found image characteristics associated with synthetic or "
                    "unusual image generation. It is not proof of AI generation."
                )


# ============================================================
# FACE SPOOF DETECTOR PAGE
# ============================================================
elif page == "👤 Face Spoof Detector":
    st.markdown("## 👤 Face Spoof / Liveness Analysis")
    st.caption(
        "Heuristic analysis for possible spoof indicators. "
        "Not intended for production biometric authentication."
    )

    col1, col2 = st.columns([1, 1])

    with col1:
        uploaded_file = st.file_uploader(
            "Upload Face Image",
            type=["png", "jpg", "jpeg", "webp"],
            key="face_upload"
        )

        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(
                image,
                caption="Uploaded Face",
                use_container_width=True
            )

    with col2:
        if uploaded_file:
            if st.button(
                "🔍 Verify Liveness",
                type="primary",
                use_container_width=True
            ):
                with st.spinner("Checking face and spoof indicators..."):
                    result, score, details, result_type, status = detect_face_spoof(image)

                log_detection(
                    "Face Spoof Detector",
                    result_type,
                    score,
                    details
                )

                if status == "real":
                    st.success(f"### {result}")
                elif status == "fake":
                    st.error(f"### {result}")
                else:
                    st.warning(f"### {result}")

                st.metric(
                    "Liveness / Spoof Confidence",
                    f"{score:.1f}%"
                )

                with st.expander(
                    "🔍 Detailed Analysis",
                    expanded=True
                ):
                    st.progress(
                        score / 100,
                        text=f"Analysis Score: {score:.1f}%"
                    )

                    st.markdown("#### 📋 Indicators")

                    for detail in details:
                        st.write(detail)


# ============================================================
# ADVERSARIAL DETECTOR PAGE
# ============================================================
elif page == "⚠️ Adversarial Detector":
    st.markdown("## ⚠️ Adversarial Image Analysis")
    st.caption(
        "Heuristic analysis of noise, gradients and frequency characteristics. "
        "It does not identify a specific attack algorithm with certainty."
    )

    col1, col2 = st.columns([1, 1])

    with col1:
        uploaded_file = st.file_uploader(
            "Upload Image",
            type=["png", "jpg", "jpeg", "webp"],
            key="adv_upload"
        )

        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(
                image,
                caption="Uploaded Image",
                use_container_width=True
            )

    with col2:
        if uploaded_file:
            if st.button(
                "⚠️ Scan for Anomalies",
                type="primary",
                use_container_width=True
            ):
                with st.spinner("Scanning image characteristics..."):
                    result, score, details, result_type, status = detect_adversarial(image)

                log_detection(
                    "Adversarial Detector",
                    result_type,
                    score,
                    details
                )

                if status == "real":
                    st.success(f"### {result}")
                elif status == "fake":
                    st.error(f"### {result}")
                else:
                    st.warning(f"### {result}")

                st.metric(
                    "Attack Analysis Score",
                    f"{score:.1f}%"
                )

                with st.expander(
                    "🔍 Detailed Analysis",
                    expanded=True
                ):
                    st.progress(
                        score / 100,
                        text=f"Analysis Score: {score:.1f}%"
                    )

                    st.markdown("#### 📋 Indicators")

                    for detail in details:
                        st.write(detail)

                st.info(
                    "ℹ️ A high score means stronger statistical anomalies "
                    "were detected; it does not by itself prove that an "
                    "FGSM, PGD or CW attack was used."
                )


# ============================================================
# SECURE CHATBOT
# ============================================================
elif page == "💬 Secure Chatbot":
    st.markdown("## 💬 Secure AI Chatbot")
    st.caption(
        "Demo prompt-security layer using keyword and phrase matching. "
        "It does not use an external LLM."
    )

    malicious_patterns = [
        "hack",
        "steal password",
        "password",
        "exploit",
        "bypass security",
        "jailbreak",
        "ignore previous",
        "ignore all previous",
        "system prompt",
        "reveal system",
        "crack",
        "malware",
        "ransomware",
        "sql injection"
    ]

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    user_input = st.chat_input(
        "Ask me about AI, cybersecurity or this project..."
    )

    if user_input:
        with st.chat_message("user"):
            st.write(user_input)

        st.session_state.chat_history.append(
            {
                "role": "user",
                "content": user_input
            }
        )

        lower_input = user_input.lower()

        is_malicious = any(
            pattern in lower_input
            for pattern in malicious_patterns
        )

        if is_malicious:
            response = (
                "🚨 **SECURITY ALERT: PROMPT BLOCKED**\n\n"
                "The message matched a potentially unsafe prompt pattern "
                "and was blocked by the demo security filter."
            )
            st.toast(
                "Potentially unsafe prompt blocked",
                icon="🚨"
            )

        elif any(
            word in lower_input.split()
            for word in ["hello", "hi", "hey"]
        ):
            response = (
                "Hello! I'm your Secure AI Assistant. "
                "I can explain AI, cybersecurity and this project."
            )

        elif "ai" in lower_input:
            response = (
                "Artificial Intelligence (AI) is a field of computing "
                "that develops systems capable of performing tasks that "
                "normally require human intelligence."
            )

        elif "security" in lower_input:
            response = (
                "Cybersecurity focuses on protecting systems, networks, "
                "applications and data from unauthorized access, misuse "
                "and digital attacks."
            )

        elif "project" in lower_input:
            response = (
                "This project demonstrates multiple security-analysis "
                "modules: AI-image heuristics, face-spoof analysis, "
                "adversarial-image analysis, prompt filtering and "
                "detection history."
            )

        elif "help" in lower_input:
            response = (
                "I can help with:\n"
                "- AI concepts\n"
                "- Cybersecurity concepts\n"
                "- Project explanation\n"
                "- Detection-module explanations\n"
                "- Basic security practices"
            )

        else:
            response = (
                "I'm a demo Secure AI Assistant. "
                "Try asking about AI, cybersecurity, the project, "
                "or how one of the detection modules works."
            )

        with st.chat_message("assistant"):
            st.write(response)

        st.session_state.chat_history.append(
            {
                "role": "assistant",
                "content": response
            }
        )


# ============================================================
# DETECTION HISTORY
# ============================================================
elif page == "📜 Detection History":
    st.markdown("## 📜 Detection History")

    if not st.session_state.history:
        st.info(
            "No detections yet. Start by scanning an image."
        )

    else:
        df = pd.DataFrame(st.session_state.history)

        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True
        )

        st.markdown("### 📊 Summary Statistics")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Total Detections",
                len(df)
            )

        with col2:
            threat_results = [
                "AI",
                "Spoof",
                "Attack",
                "Suspicious",
                "NoFace"
            ]

            threats = df[
                df["Result"].isin(threat_results)
            ].shape[0]

            st.metric(
                "Threat / Warning Results",
                threats
            )

        with col3:
            safe = df[
                df["Result"].isin(["Real", "Clean"])
            ].shape[0]

            st.metric(
                "Safe / Clean Results",
                safe
            )

        csv_data = df.to_csv(index=False)

        st.download_button(
            "📥 Download History as CSV",
            data=csv_data,
            file_name="detection_history.csv",
            mime="text/csv"
        )

        st.markdown("### 🗑️ Manage History")

        if st.button(
            "Clear Detection History",
            type="secondary"
        ):
            st.session_state.history = []
            st.session_state.total_scans = 0
            st.session_state.threats = 0
            st.rerun()


# ============================================================
# FOOTER
# ============================================================
st.markdown("---")

st.markdown(
    """
    <div style="text-align:center;">
        <b>Secure AI Model Defense System</b>
        | B.Tech CSE Project
        | Simran Kumari
        <br>
        Multi-layer security analysis dashboard
    </div>
    """,
    unsafe_allow_html=True
)
