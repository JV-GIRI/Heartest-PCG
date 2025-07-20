import streamlit as st
import numpy as np, scipy.signal as signal, librosa
from scipy.signal import find_peaks
from fpdf import FPDF
import tempfile

st.title("💓 Heartest: Murmur Detector")

uploaded = st.file_uploader("Upload PCG .wav", type=["wav"])
if uploaded:
    y, sr = librosa.load(uploaded, sr=None)
    # Preprocess
    y2 = signal.savgol_filter(signal.filtfilt(
        *signal.butter(4,[20/(sr/2),500/(sr/2)],'band'), np.clip(y*3,-1,1)
    ),501,3)
    # HR & murmur
    peaks, _ = find_peaks(np.abs(signal.hilbert(y2)), distance=sr*0.4)
    hr = 60 / np.mean(np.diff(peaks)/sr) if len(peaks)>1 else 0
    S = librosa.feature.melspectrogram(y=y2, sr=sr, n_mels=40)
    murmur = np.mean(librosa.power_to_db(S, ref=np.max)[10:30]) > -35
    # Show results
    st.write(f"**Heart Rate:** {hr:.1f} bpm")
    st.write("🔴 Murmur Detected!" if murmur else "🟢 No Murmur Detected")
    # Generate PDF
    if st.button("📥 Download Report"):
        text = f"Heart Rate: {hr:.1f} bpm\n"
        text += "Murmur: YES (possible valvular disease)\n" if murmur else "Murmur: NO\n"
        pdf = FPDF(); pdf.add_page(); pdf.set_font("Arial",12)
        for line in text.split("\n"):
            pdf.cell(0,10,line,ln=True)
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        pdf.output(tmp.name)
        st.download_button("Download PDF", data=open(tmp.name,'rb'),
                           file_name="Heartest_Report.pdf", mime="application/pdf")
