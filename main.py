import cv2
from Handtracking_module import HandDetector
import numpy as np
import google.generativeai as genai
import os
from PIL import Image
import streamlit as st
import time
from textwrap import dedent


st.set_page_config(page_title="Math Gesture AI", page_icon="✋", layout="wide")

# --- Hero section ---
st.title("Math Gesture AI")
st.caption("Hands-free math solving with air-drawn gestures and Gemini free tier models.")

intro_col1, intro_col2 = st.columns([2, 3])
with intro_col2:
    st.markdown(
        dedent(
            """
            **How to use**
            - Raise your index finger to draw strokes in mid-air.
            - Raise only your thumb to clear the virtual canvas.
            - Hold all five fingers steady to submit the captured gesture to Gemini.
            """
        )
    )
with intro_col1:
    st.metric("Gesture Trigger", "Five fingers up")
    st.metric("Cool-down", "15 seconds")
    st.metric("Model Tier", "Gemini Free")
# Sidebar controls
with st.sidebar:
    st.header("Control Panel")
    run = st.checkbox("Live tracking", value=True)
    clear_canvas_requested = st.button("🧼 Clear Canvas")

    st.markdown("---")
    st.subheader("Deploy")
    st.code("streamlit run main.py", language="bash")
    st.markdown(
        dedent(
            """
            1. Create `.streamlit/secrets.toml` with `GEMINI_API_KEY`.
            2. Install dependencies: `pip install -r requirements.txt`.
            3. Run locally or deploy via Streamlit Community Cloud.
            """
        )
    )

    st.markdown("---")
    st.subheader("GitHub")
    st.markdown("[View repository]https://github.com/ruhul-cse-duet/math_gesture_ai_streamlit.git)")
    st.markdown("Commit message suggestion: `feat: polish UI for Math Gesture AI`")

st.markdown("---")

# Layout for live feed and answer
visual_col, answer_col = st.columns([2, 3])

with answer_col:
    st.subheader("Live Feed")
    FRAME_WINDOW = st.empty()

with visual_col:
    st.subheader("AI Solution")
    status_placeholder = st.empty()
    output_placeholder = st.empty()
    status_placeholder.info("Hold five fingers steady to submit your gesture to Gemini.")

if not run:
    status_placeholder.warning("Enable 'Live tracking' from the sidebar to start the webcam feed.")

tabs = st.tabs(["How it works", "Deploy", "GitHub project"])
with tabs[0]:
    st.markdown(
        dedent(
            """
            1. The webcam feed is processed with MediaPipe hands via `HandDetector`.
            2. Detected landmarks drive a virtual canvas that mirrors your air-drawn strokes.
            3. Holding all five fingers steady triggers a canvas snapshot to Gemini for solving.
            4. The model response streams back into the **AI Solution** panel.
            """
        )
    )

with tabs[1]:
    st.markdown(
        dedent(
            """
            **Local run**
            - `python -m venv .venv && .venv\\Scripts\\activate`
            - `pip install -r requirements.txt`
            - `streamlit run main.py`

            **Streamlit Community Cloud**
            - Connect your GitHub repo (main branch).
            - Set the main file to `main.py` and Python version to 3.10+.
            - Add `GEMINI_API_KEY` under App secrets.
            - Redeploy after updating the repo to auto-refresh.
            """
        )
    )

with tabs[2]:
    st.markdown(
        dedent(
            """
            - Recommended repository name: `math-gesture-ai`
            - Key files:
                - `main.py` – Streamlit UI & inference loop
                - `Handtracking_module.py` – hand landmark utilities
                - `requirements.txt` – dependencies for deployment
                - `.streamlit/secrets.toml.example` – sample secrets template
            - Add GitHub Actions to lint and format (e.g., `flake8`, `black`).
            - Showcase demo GIFs and deployment badges in `README.md`.
            """
        )
    )

# Configure API key, prefer Streamlit secrets or environment variable for free-tier keys
api_key = None
try:
    api_key = st.secrets.get("AIzaSyArv4fSsKICxhvP6Cw7IFcyjeALemVa-Sk", None)
except Exception:
    api_key = None
if not api_key:
    api_key = os.environ.get("AIzaSyArv4fSsKICxhvP6Cw7IFcyjeALemVa-Sk")
if not api_key:
    # fallback to hardcoded (not recommended)
    api_key = "AIzaSyArv4fSsKICxhvP6Cw7IFcyjeALemVa-Sk"

genai.configure(api_key=api_key)

# Use free-tier friendly model directly
PREFERRED_MODELS = [
    # 1.5 Flash family (fast, usually free tier)
    "gemini-2.5-flash",
    "gemini-2.0-flash-exp",
    "gemini-1.5-flash",
    "gemini-1.5-flash-latest",
    "gemini-1.5-flash-001",
    "gemini-1.5-flash-8b",
    # Vision-capable legacy/free models for v1beta projects
    "gemini-pro-vision",
    "gemini-1.0-pro-vision",
    # As a last resort (text-only; may fail with image input)
    "gemini-pro",
    "gemini-1.0-pro"
]

model = None
for model_name in PREFERRED_MODELS:
    try:
        model = genai.GenerativeModel(model_name=model_name)
        st.caption(f"Using model: {model_name}")
        break
    except Exception:
        continue
if model is None:
    st.error("Could not initialize a free Gemini model. Check your API key access or enable a Gemini Flash/Pro Vision model.")
    st.stop()

# Initialize the webcam to capture video
# Try the provided index; if unavailable, the loop below will avoid crashes
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

# Fail fast if the camera can't be opened
if not cap.isOpened():
    st.error("Camera index 0 could not be opened. Close other apps or try a different index (e.g., 1).")
    st.stop()

# Initialize the HandDetector class with the given parameters
detector = HandDetector(staticMode=False, maxHands=1, modelComplexity=1, detectionCon=0.7, minTrackCon=0.5)


def getHandInfo(img):
    # Find hands in the current frame
    # The 'draw' parameter draws landmarks and hand outlines on the image if set to True
    # The 'flipType' parameter flips the image, making it easier for some detections
    hands, img = detector.findHands(img, draw=False, flipType=True)

    # Check if any hands are detected
    if hands:
        # Information for the first hand detected
        hand = hands[0]  # Get the first hand detected
        lmList = hand["lmList"]  # List of 21 landmarks for the first hand
        # Count the number of fingers up for the first hand
        fingers = detector.fingersUp(hand)
        #print(fingers)
        return fingers, lmList
    else:
        return None


def draw(info, prev_pos, canvas):
    fingers, lmList = info
    current_pos = None
    if fingers == [0, 1, 0, 0, 0]:
        current_pos = lmList[8][0:2]
        if prev_pos is None: prev_pos = current_pos
        cv2.line(canvas, current_pos, prev_pos, (255, 0, 255), 10)
    elif fingers == [1, 0, 0, 0, 0]:
        canvas = np.zeros_like(img)

    return current_pos, canvas


def sendToAI(model, canvas, fingers, status_placeholder, max_retries=2):
    if fingers == [1, 1, 1, 1, 1]:
        backoff_seconds = 2
        attempt = 0
        while attempt <= max_retries:
            try:
                pil_image = Image.fromarray(canvas)
                response = model.generate_content(["Solve this math problem", pil_image])
                status_placeholder.success("Gemini completed the request")
                return getattr(response, "text", None)
            except Exception as exc:
                message = str(exc)
                if "429" in message and attempt < max_retries:
                    status_placeholder.warning(f"Rate limited. Retrying in {backoff_seconds}s...")
                    time.sleep(backoff_seconds)
                    backoff_seconds *= 2
                    attempt += 1
                    continue
                status_placeholder.error(f"AI request failed: {exc}")
                return None


prev_pos = None
canvas = None
image_combined = None
output_text = ""

# Gesture gating and cooldown to avoid spamming the API
consecutive_all_fingers = 0
required_frames_for_trigger = 8  # ~8 frames of stability before triggering
last_request_time = 0.0
cooldown_seconds = 15  # minimum gap between AI calls
# Continuously get frames from the webcam
while run:
    # Capture each frame from the webcam
    # 'success' will be True if the frame is successfully captured, 'img' will contain the frame
    success, img = cap.read()
    if not success or img is None:
        # Skip this iteration if frame grab failed
        continue
    img = cv2.flip(img, 1)

    if canvas is None:
        canvas = np.zeros_like(img)

    if clear_canvas_requested:
        canvas = np.zeros_like(img)
        prev_pos = None
        output_text = ""
        status_placeholder.info("Canvas cleared. Draw a new expression.")
        clear_canvas_requested = False

    info = getHandInfo(img)
    if info:
        fingers, lmList = info
        prev_pos, canvas = draw(info, prev_pos, canvas)

        # Count stable frames of five-finger gesture
        if fingers == [1, 1, 1, 1, 1]:
            consecutive_all_fingers += 1
        else:
            consecutive_all_fingers = 0

        # Trigger only when stable and after cooldown
        now = time.time()
        if consecutive_all_fingers >= required_frames_for_trigger and (now - last_request_time) >= cooldown_seconds:
            status_placeholder.info("Processing with Gemini...")
            ai_result = sendToAI(model, canvas, fingers, status_placeholder=status_placeholder)
            last_request_time = now
            if ai_result:
                output_text = ai_result
            consecutive_all_fingers = 0

    image_combined = cv2.addWeighted(img, 0.7, canvas, 0.3, 0)
    FRAME_WINDOW.image(image_combined, channels="BGR")

    if output_text:
        output_placeholder.subheader(output_text)
    else:
        output_placeholder.write("Awaiting a submitted gesture ✋✨")

    # # Display the image in a window
    # cv2.imshow("Image", img)
    # cv2.imshow("Canvas", canvas)
    # cv2.imshow("image_combined", image_combined)

    # Keep the window open and update it for each frame; wait for 1 millisecond between frames
    cv2.waitKey(1)