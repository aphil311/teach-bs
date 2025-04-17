import streamlit as st
import numpy as np
import pandas as pd

from modules.nav import Navbar

Navbar()

# Set page title
st.title("Score Analysis")

# Initialize session state for scores if it doesn't exist
if 'scores' not in st.session_state:
    st.session_state.scores = []

# Display scores if they exist
if st.session_state.scores:
    # Calculate average
    average_score = np.mean(st.session_state.scores)

    if average_score > 90:
        st.balloons()
    
    # Display the average
    st.header("Score Results")
    st.metric(label="Average Score", value=f"{average_score:.2f}")
else:
    st.info("No scores have been entered yet. Please chat with the bot first!")