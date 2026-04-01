import streamlit as st
import os

st.title("🚨 AI Surveillance Dashboard")

st.subheader("📄 Intruder Logs")

log_file = "logs/log.txt"

if os.path.exists(log_file):
    with open(log_file, "r") as f:
        logs = f.readlines()

    for log in logs[::-1]:
        st.write(log)
else:
    st.write("No logs available")

st.subheader("📸 Captured Intruder Images")

if os.path.exists("logs"):
    images = os.listdir("logs")

    for img in images:
        if img.endswith(".jpg"):
            st.image(f"logs/{img}", caption=img)
else:
    st.write("No images found")
