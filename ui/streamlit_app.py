"""Upload / webcam QC demo."""

import streamlit as st

st.title("Manufacturing QC — YOLOv8")
st.file_uploader("Upload product image", type=["jpg", "png", "jpeg"])
