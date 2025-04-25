import streamlit as st


def Navbar():
    with st.sidebar:
        st.title("Der Roboterlehrer")
        st.markdown("### Translation Bots")
        st.page_link("app.py", label="German to English", icon="🇩🇪")
        st.page_link("pages/to_german.py", label="English to German", icon="🇬🇧")
        st.markdown("### Analysis")
        st.page_link("pages/scoring.py", label="Score Analysis", icon="📊")
        st.divider()
        st.markdown("### About")
        st.markdown(
            "This app is a translation bot that helps you practice your language skills. It uses machine learning models to evaluate your translations and provide feedback."
        )
