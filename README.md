Math Gesture AI
================

Hands-free math solving with air-drawn gestures, computer vision, and Gemini. The app uses a webcam to track your hand, lets you draw equations in the air, and forwards the captured canvas to a Gemini free-tier model for instant problem solving — all wrapped in a Streamlit interface ready for deployment.

Features
--------

- Real-time hand landmark detection via MediaPipe (through `HandDetector`).
- Air drawing on a virtual canvas with index finger, plus gesture-based clearing.
- Gesture gating and cooldown to avoid API overuse.
- Streamlit UI with live feed preview and AI response panel.
- Built-in deployment guidance for local runs and Streamlit Community Cloud.

Getting Started
---------------

### Prerequisites

- Python 3.10 or newer
- Webcam
- Google Gemini API key (MakerSuite or Google AI Studio)

### Installation

```bash
git clone https://github.com/USERNAME/Math-Gesture-AI.git
cd Math-Gesture-AI
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # macOS/Linux
pip install -r requirements.txt
```

Create `.streamlit/secrets.toml` and add your API key:

```toml
GEMINI_API_KEY = "your_api_key_here"
```

Usage
-----

Run the Streamlit app:

```bash
streamlit run main.py
```

Open the displayed URL in your browser. In the sidebar, toggle `Live tracking` to start the webcam feed. Hold five fingers steady to trigger Gemini.

Gesture Guide
-------------

- Index finger up: draw on the canvas
- Thumb up only: clear the canvas
- All five fingers steady: submit to Gemini

Deployment
----------

### Streamlit Community Cloud

1. Push this project to GitHub (main branch recommended).
2. Create an app on [share.streamlit.io](https://share.streamlit.io).
3. Point to `main.py` and set Python 3.10+.
4. Add the `GEMINI_API_KEY` secret in the Streamlit dashboard.
5. Deploy — updates are pulled automatically when you push new commits.

### Other Platforms

- **Docker**: Build an image with `streamlit` entry point.
- **Local server**: Wrap `streamlit run main.py` in a systemd service or Windows Task Scheduler task.

Project Structure
-----------------

```
Math Gesture AI/
├── main.py                  # Streamlit UI and inference loop
├── Handtracking_module.py   # MediaPipe hand detection helper
├── README.md
├── requirements.txt         # Python dependencies
└── .streamlit/
    └── secrets.toml.example # Sample secrets file (add your own)
```

Roadmap & Ideas
---------------

- Add support for more complex math symbols (integrals, summations).
- Provide a history panel of solved expressions.
- Export captured gestures as SVG/PNG snapshots.
- Integrate alternative models (local OCR, Claude, etc.) via a selector.

Contributing
------------

Issues and pull requests are welcome. Before contributing, please:

1. Open an issue describing the enhancement or bug fix.
2. Fork the repository and create a feature branch.
3. Run linting/tests before submitting a pull request.

License
-------

Specify your chosen license here (e.g., MIT). If the project remains private, remove this section.

