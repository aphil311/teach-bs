import time

import streamlit as st

from categories.accuracy import *


def response_generator(prompt):
    source = st.session_state.german
    acc = accuracy(source, prompt)

    response = "Your response is: " + str(acc["score"]) + "\n"

    if acc["errors"]:
        response += "Your errors are:\n"

        for error in acc["errors"]:
            response += f" - {error['message']}\n"

    lines = response.split("\n")
    for line in lines:
        for word in line.split():
            yield word + " "
            time.sleep(0.05)
        # After each line, yield a newline character or trigger a line break in Markdown
        yield "\n"


def translation_generator():
    st.session_state.german = "Danke shoen."

    message = (
        f"Please translate the following sentence into English:"
        f" {st.session_state.german}"
    )

    lines = message.split("\n")
    for line in lines:
        for word in line.split():
            yield word + " "
            time.sleep(0.05)
        # After each line, yield a newline character or trigger a line break in Markdown
        yield "\n"


st.title("Translation bot")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Hello! I am a translation bot. Please translate the following"
                " sentence into English: 'Das ist ein Test.'"
            ),
        }
    ]
    st.session_state.german = "Das ist ein Test."

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What is up?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        response = st.write_stream(response_generator(prompt))

    st.session_state.messages.append({"role": "assistant", "content": response})

    with st.chat_message("assistant"):
        message = st.write_stream(translation_generator())

    st.session_state.messages.append({"role": "assistant", "content": message})
    # Add assistant response to chat history
