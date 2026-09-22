import base64
import io
import json
from pathlib import Path
import subprocess
import tempfile

import numpy as np
from scipy.io import wavfile
import streamlit as st

from main import read_audio, recognize_stems


def analyze(data, progress, **parameters):
    with tempfile.NamedTemporaryFile() as file:
        file.write(data)
        file.flush()
        sample_rate, audio = read_audio(file.name)
    labels = recognize_stems(audio, sample_rate, progress=progress, **parameters)
    wav = io.BytesIO()
    wavfile.write(wav, sample_rate, (np.clip(audio, -1, 1) * 32767).astype(np.int16))
    frame_seconds = round(sample_rate * 0.01) * parameters["frames_per_group"] / sample_rate
    return wav.getvalue(), labels.tolist(), frame_seconds, int(labels.max(initial=-1)) + 1


st.title("Stems recognition")
upload = st.file_uploader("Drop or choose music", type=["mp3", "wav", "flac", "ogg", "m4a"],
                          on_change=lambda: st.session_state.pop("result", None))
defaults = dict(frames_per_group=2, n_bands=100, min_cluster_size=10, min_samples=10, split_size=0.2, cluster_selection_epsilon=0.0, alpha=1.0, power=0.5)
with st.form("parameters"):
    fields = {name: st.text_input(name, str(value)) for name, value in defaults.items()}
    algorithm = st.selectbox("algorithm", ["auto", "brute", "kd_tree", "ball_tree"])
    metric = st.selectbox("metric", ["euclidean", "manhattan", "cosine", "chebyshev", "correlation"],
                          help="Cosine and correlation require algorithm auto or brute.")
    submitted = st.form_submit_button("Analyze audio", disabled=upload is None)
if submitted:
    try:
        parameters = {name: type(defaults[name])(value) for name, value in fields.items()}
        parameters["algorithm"] = algorithm
        parameters["metric"] = metric
        bar = st.progress(0, text="Reading audio…")
        result = analyze(upload.getvalue(), bar.progress, **parameters)
        st.session_state.result = result
    except (subprocess.CalledProcessError, ValueError) as error:
        st.error(f"Could not analyze this audio: {error}")
        st.stop()
if "result" in st.session_state:
    wav, labels, frame_seconds, n_clusters = st.session_state.result
    st.write(f"{n_clusters} clusters · {labels.count(-1)} noise frames")
    player = Path(__file__).with_name("player.html").read_text()
    player = player.replace("__AUDIO__", base64.b64encode(wav).decode())
    player = player.replace("__LABELS__", json.dumps(labels))
    player = player.replace("__N_CLUSTERS__", json.dumps(n_clusters))
    player = player.replace("__FRAME_SECONDS__", json.dumps(frame_seconds))
    st.iframe(player, height=260)
