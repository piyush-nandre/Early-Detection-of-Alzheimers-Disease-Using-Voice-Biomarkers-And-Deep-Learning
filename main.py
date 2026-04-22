import os
import tempfile

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import tensorflow as tf

# ---------------------------
# CONFIG
# ---------------------------
MODEL_PATH = "alz_cnn.keras"
N_MFCC = 40
MAX_PAD = 250
SR_TARGET = 16000
THRESHOLD = 0.5


st.set_page_config(
    page_title="NeuroVoice Screen",
    page_icon="N",
    layout="wide",
)


def inject_styles():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Space+Grotesk:wght@500;700&display=swap');

        :root {
            --bg: #f4efe6;
            --surface: rgba(255, 252, 246, 0.82);
            --surface-strong: rgba(255, 248, 238, 0.95);
            --border: rgba(84, 60, 44, 0.10);
            --ink: #21160f;
            --muted: #6b5445;
            --accent: #c96f3b;
            --accent-soft: #f0b489;
            --success: #2f7d62;
            --danger: #b94b52;
            --shadow: 0 24px 60px rgba(75, 47, 25, 0.10);
        }

        .stApp {
            background:
                radial-gradient(circle at top left, rgba(255, 213, 179, 0.65), transparent 28%),
                radial-gradient(circle at 85% 10%, rgba(227, 205, 177, 0.85), transparent 22%),
                linear-gradient(180deg, #f9f3ea 0%, var(--bg) 45%, #efe6da 100%);
            color: var(--ink);
            font-family: 'Outfit', sans-serif;
        }

        .block-container {
            padding-top: 2.2rem;
            padding-bottom: 3rem;
            max-width: 1180px;
        }

        h1, h2, h3, h4 {
            color: var(--ink);
            font-family: 'Space Grotesk', sans-serif;
            letter-spacing: -0.02em;
        }

        p, li, label, div, span {
            font-family: 'Outfit', sans-serif;
        }

        [data-testid="stHeader"] {
            background: rgba(0, 0, 0, 0);
        }

        [data-testid="stFileUploader"] {
            background: rgba(255, 255, 255, 0.36);
            border: 1px dashed rgba(107, 84, 69, 0.35);
            border-radius: 24px;
            padding: 1rem;
        }

        [data-testid="stMetric"] {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 22px;
            padding: 0.75rem;
            box-shadow: var(--shadow);
        }

        .stButton > button, .stDownloadButton > button {
            background: linear-gradient(135deg, #cf6f3e 0%, #df8d58 100%);
            color: #fffaf4;
            border: none;
            border-radius: 999px;
            padding: 0.75rem 1.4rem;
            font-weight: 600;
            box-shadow: 0 12px 24px rgba(201, 111, 59, 0.25);
        }

        .stButton > button:hover, .stDownloadButton > button:hover {
            background: linear-gradient(135deg, #bb6132 0%, #d47f4d 100%);
        }

        .hero {
            position: relative;
            overflow: hidden;
            background: linear-gradient(145deg, rgba(255, 252, 246, 0.94), rgba(247, 233, 216, 0.92));
            border: 1px solid var(--border);
            border-radius: 32px;
            padding: 2.8rem;
            box-shadow: var(--shadow);
            margin-bottom: 1.5rem;
        }

        .hero::after {
            content: "";
            position: absolute;
            inset: auto -60px -60px auto;
            width: 220px;
            height: 220px;
            background: radial-gradient(circle, rgba(201, 111, 59, 0.16), transparent 65%);
        }

        .eyebrow {
            display: inline-block;
            padding: 0.4rem 0.85rem;
            border-radius: 999px;
            background: rgba(201, 111, 59, 0.10);
            color: #a15329;
            font-size: 0.82rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-bottom: 1rem;
        }

        .hero-grid {
            display: grid;
            grid-template-columns: 1.4fr 0.9fr;
            gap: 1.2rem;
            align-items: end;
        }

        .hero-copy h1 {
            font-size: clamp(2.2rem, 4vw, 4rem);
            line-height: 1.02;
            margin: 0 0 0.8rem 0;
        }

        .hero-copy p {
            color: var(--muted);
            font-size: 1.02rem;
            line-height: 1.75;
            max-width: 58ch;
            margin-bottom: 0;
        }

        .stat-panel {
            background: rgba(255, 248, 239, 0.88);
            border: 1px solid rgba(107, 84, 69, 0.12);
            border-radius: 24px;
            padding: 1.25rem;
            backdrop-filter: blur(14px);
        }

        .stat-kicker {
            color: var(--muted);
            font-size: 0.88rem;
            margin-bottom: 0.45rem;
        }

        .stat-value {
            font-family: 'Space Grotesk', sans-serif;
            font-size: 2.25rem;
            font-weight: 700;
            margin-bottom: 0.4rem;
        }

        .stat-note {
            color: var(--muted);
            line-height: 1.6;
            font-size: 0.94rem;
        }

        .section-card {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 28px;
            padding: 1.4rem;
            box-shadow: var(--shadow);
            margin-bottom: 1.2rem;
            backdrop-filter: blur(16px);
        }

        .section-title {
            font-family: 'Space Grotesk', sans-serif;
            font-size: 1.2rem;
            font-weight: 700;
            margin-bottom: 0.35rem;
        }

        .section-subtitle {
            color: var(--muted);
            line-height: 1.65;
            margin-bottom: 1rem;
        }

        .mini-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.8rem;
            margin-top: 1rem;
        }

        .mini-card {
            background: rgba(255, 255, 255, 0.56);
            border: 1px solid rgba(107, 84, 69, 0.10);
            border-radius: 20px;
            padding: 1rem;
        }

        .mini-card-label {
            color: var(--muted);
            font-size: 0.82rem;
            margin-bottom: 0.25rem;
        }

        .mini-card-value {
            font-family: 'Space Grotesk', sans-serif;
            font-size: 1.2rem;
            font-weight: 700;
        }

        .result-banner {
            border-radius: 28px;
            padding: 1.35rem;
            margin-bottom: 1rem;
            border: 1px solid transparent;
        }

        .result-banner.safe {
            background: linear-gradient(135deg, rgba(47, 125, 98, 0.14), rgba(212, 243, 226, 0.58));
            border-color: rgba(47, 125, 98, 0.18);
        }

        .result-banner.alert {
            background: linear-gradient(135deg, rgba(185, 75, 82, 0.12), rgba(255, 223, 219, 0.62));
            border-color: rgba(185, 75, 82, 0.16);
        }

        .result-tag {
            display: inline-block;
            font-size: 0.76rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            padding: 0.35rem 0.7rem;
            border-radius: 999px;
            margin-bottom: 0.75rem;
            background: rgba(255, 255, 255, 0.55);
        }

        .result-title {
            font-family: 'Space Grotesk', sans-serif;
            font-size: 1.55rem;
            font-weight: 700;
            margin-bottom: 0.3rem;
        }

        .result-copy {
            color: var(--muted);
            line-height: 1.7;
            margin-bottom: 0;
        }

        .disclaimer {
            background: rgba(255, 251, 246, 0.72);
            border: 1px solid rgba(107, 84, 69, 0.12);
            color: var(--muted);
            border-radius: 22px;
            padding: 1rem 1.1rem;
            line-height: 1.7;
            margin-top: 1.2rem;
        }

        @media (max-width: 900px) {
            .hero-grid, .mini-grid {
                grid-template-columns: 1fr;
            }

            .hero {
                padding: 1.5rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource
def load_cnn_model(path: str):
    if not os.path.exists(path):
        st.error(f"Model file `{path}` not found in this folder.")
        st.stop()

    try:
        return tf.keras.models.load_model(path, compile=False)
    except Exception as exc:
        st.error(
            "Error loading model.\n\n"
            "This usually happens because the TensorFlow or Keras version used "
            "to save the model does not match the current environment.\n\n"
            f"Technical details:\n`{exc}`"
        )
        st.stop()


def extract_mfcc_cnn(file_path: str):
    y, sr = librosa.load(file_path, sr=SR_TARGET)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)

    if mfcc.shape[1] < MAX_PAD:
        pad_width = MAX_PAD - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode="constant")
    else:
        mfcc = mfcc[:, :MAX_PAD]

    mfcc_input = mfcc.reshape(1, N_MFCC, MAX_PAD, 1)
    return y, sr, mfcc, mfcc_input


def run_inference(audio_path: str, model):
    y, sr, mfcc_orig, mfcc_ready = extract_mfcc_cnn(audio_path)
    prob_diseased = float(model.predict(mfcc_ready, verbose=0)[0][0])
    risk_score = prob_diseased * 100

    if prob_diseased >= THRESHOLD:
        label = "Patient-like speech pattern detected"
        category = "Higher-risk screening result"
        tone = "alert"
    else:
        label = "Healthy-like speech pattern detected"
        category = "Lower-risk screening result"
        tone = "safe"

    return y, sr, mfcc_orig, prob_diseased, risk_score, label, category, tone


def create_waveform_figure(y, sr):
    fig, ax = plt.subplots(figsize=(10, 3.2), facecolor="#fffaf4")
    ax.set_facecolor("#fffaf4")
    librosa.display.waveshow(y, sr=sr, ax=ax, color="#c96f3b", alpha=0.9)
    ax.set_title("Voice waveform", fontsize=14, fontweight="bold", color="#21160f")
    ax.tick_params(colors="#6b5445")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(alpha=0.15, color="#6b5445")
    fig.tight_layout()
    return fig


def create_mfcc_figure(mfcc, sr):
    fig, ax = plt.subplots(figsize=(10, 4), facecolor="#fffaf4")
    ax.set_facecolor("#fffaf4")
    img = librosa.display.specshow(
        mfcc,
        x_axis="time",
        sr=sr,
        ax=ax,
        cmap="copper",
    )
    ax.set_title("MFCC feature map", fontsize=14, fontweight="bold", color="#21160f")
    ax.tick_params(colors="#6b5445")
    for spine in ax.spines.values():
        spine.set_visible(False)
    cbar = fig.colorbar(img, ax=ax, pad=0.02)
    cbar.ax.tick_params(colors="#6b5445")
    fig.tight_layout()
    return fig


def render_hero():
    st.markdown(
        """
        <section class="hero">
            <div class="hero-grid">
                <div class="hero-copy">
                    <div class="eyebrow">Voice Biomarker Interface</div>
                    <h1>AI screening for early cognitive and dementia voice signals.</h1>
                    <p>
                        Upload a short speech sample and review a refined visual report
                        built around your model's dementia-like versus healthy-like screening output.
                    </p>
                </div>
                <div class="stat-panel">
                    <div class="stat-kicker">Model input profile</div>
                    <div class="stat-value">40 MFCC x 250 frames</div>
                    <div class="stat-note">
                        The interface is tuned for calm readability, stronger hierarchy,
                        and a cleaner presentation of audio insights and screening results.
                    </div>
                </div>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_input_panel():
    st.markdown(
        """
        <section class="section-card">
            <div class="section-title">Upload Speech Sample</div>
            <div class="section-subtitle">
                Use a clear voice recording in <strong>.wav</strong>, <strong>.flac</strong>,
                <strong>.mp3</strong>, or <strong>.m4a</strong> format, or record directly in the app.
                The sample will be processed, converted into MFCC features, and passed to the CNN model.
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_results(prob_diseased, risk_score, label, category, tone, y, sr, mfcc):
    banner_class = "alert" if tone == "alert" else "safe"
    confidence_gap = abs(prob_diseased - THRESHOLD)
    confidence = "High" if confidence_gap >= 0.30 else "Moderate" if confidence_gap >= 0.15 else "Low"
    audio_length = len(y) / sr

    st.markdown(
        f"""
        <section class="result-banner {banner_class}">
            <div class="result-tag">{category}</div>
            <div class="result-title">{label}</div>
            <p class="result-copy">
                Screening probability for dementia-like speech is <strong>{prob_diseased:.3f}</strong>
                using a decision threshold of <strong>{THRESHOLD:.2f}</strong>.
            </p>
        </section>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Risk score", f"{risk_score:.1f}%")
    col2.metric("Confidence", confidence)
    col3.metric("Audio length", f"{audio_length:.1f}s")

    st.markdown(
        """
        <section class="section-card">
            <div class="section-title">Screening Snapshot</div>
            <div class="section-subtitle">
                A compact view of the main decision signals extracted from the current upload.
            </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
            <div class="mini-grid">
                <div class="mini-card">
                    <div class="mini-card-label">Decision threshold</div>
                    <div class="mini-card-value">{THRESHOLD:.2f}</div>
                </div>
                <div class="mini-card">
                    <div class="mini-card-label">MFCC coefficients</div>
                    <div class="mini-card-value">{N_MFCC}</div>
                </div>
                <div class="mini-card">
                    <div class="mini-card-label">Target sample rate</div>
                    <div class="mini-card-value">{SR_TARGET:,} Hz</div>
                </div>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    left, right = st.columns([1.05, 1])
    
    
    with left:
        st.markdown(
            """
            <section class="section-card">
                <div class="section-title">Waveform Preview</div>
                <div class="section-subtitle">
                    Temporal amplitude view of the uploaded speech sample.
                </div>
            """,
            unsafe_allow_html=True,
        )
        st.pyplot(create_waveform_figure(y, sr), use_container_width=True)
        st.markdown("</section>", unsafe_allow_html=True)

    with right:
        st.markdown(
            """
            <section class="section-card">
                <div class="section-title">MFCC Spectrogram</div>
                <div class="section-subtitle">
                    Feature map used as the CNN input after padding or cropping.
                </div>
            """,
            unsafe_allow_html=True,
        )
        st.pyplot(create_mfcc_figure(mfcc, sr), use_container_width=True)
        st.markdown("</section>", unsafe_allow_html=True)


def main():
    inject_styles()
    model = load_cnn_model(MODEL_PATH)

    render_hero()
    render_input_panel()

    col1, col2 = st.columns([1.2, 0.8])
    with col1:
        input_mode = st.radio(
            "Choose input method",
            ["Upload audio", "Record now"],
            horizontal=True,
        )

        uploaded = None
        recorded = None

        if input_mode == "Upload audio":
            uploaded = st.file_uploader(
                "Drop in a voice sample",
                type=["wav", "flac", "mp3", "m4a"],
                label_visibility="collapsed",
            )
        else:
            recorded = st.audio_input("Record a voice sample")

    with col2:
        st.markdown(
            """
            <section class="section-card">
                <div class="section-title">How to use</div>
                <div class="section-subtitle">
                    Choose upload or live recording, preview the sample, and review the model output with waveform and MFCC views.
                </div>
                <div class="mini-grid">
                    <div class="mini-card">
                        <div class="mini-card-label">Step 1</div>
                        <div class="mini-card-value">Capture</div>
                    </div>
                    <div class="mini-card">
                        <div class="mini-card-label">Step 2</div>
                        <div class="mini-card-value">Analyze</div>
                    </div>
                    <div class="mini-card">
                        <div class="mini-card-label">Step 3</div>
                        <div class="mini-card-value">Review</div>
                    </div>
                </div>
            </section>
            """,
            unsafe_allow_html=True,
        )

    audio_source = uploaded if uploaded is not None else recorded

    if audio_source is None:
        st.markdown(
            """
            <div class="disclaimer">
                Upload a speech file or record one on the spot to begin. This interface is designed
                for academic screening workflows and should not be treated as a medical diagnosis.
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    st.markdown(
        """
        <section class="section-card">
            <div class="section-title">Audio Preview</div>
            <div class="section-subtitle">
                Listen back to the selected sample before reviewing the generated output.
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )
    st.audio(audio_source)

    source_name = getattr(audio_source, "name", "recorded_audio.wav")
    suffix = os.path.splitext(source_name)[1].lower() or ".wav"
    tmp_path = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(audio_source.getbuffer())
            tmp_path = tmp.name

        with st.spinner("Processing audio and generating the screening report..."):
            y, sr, mfcc_orig, prob_diseased, risk_score, label, category, tone = run_inference(tmp_path, model)

        render_results(prob_diseased, risk_score, label, category, tone, y, sr, mfcc_orig)

    except Exception as exc:
        st.error(
            "Error while processing this audio file.\n\n"
            "Common reasons include a missing backend for non-WAV files, an unsupported file, "
            "or a corrupted upload.\n\n"
            f"Technical details:\n`{exc}`"
        )
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)

    st.markdown(
        """
        <div class="disclaimer">
            This tool is intended for academic and research screening support only.
            It should not be used as a standalone clinical diagnosis for Alzheimer's disease or dementia.
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
