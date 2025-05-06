import json
import random
import time

import streamlit as st

from categories.accuracy import *
from categories.fluency import *
from categories.style import *
from modules.nav import Navbar

Navbar()


# Load translations from a JSON file to be used by the bot
def load_translations():
    try:
        with open("./data/translations.json", "r") as f:
            return json.loads(f.read())
    except Exception as e:
        print(e)
        return None


def score(score):
    if score > 90:
        return "excellent!"
    elif score > 70:
        return "good."
    elif score > 50:
        return "fair."
    else:
        return "poor."


if "translations" not in st.session_state:
    st.session_state.translations = load_translations()


def response_generator(prompt):
    source = st.session_state.german
    acc = accuracy(source, prompt)
    ppl = pseudo_perplexity(prompt)
    gre = grammar_errors(prompt)
    frm = formality(source, prompt)

    total_score = (
        0.5 * acc["score"]
        + 0.2 * gre["score"]
        + 0.3 * ppl["score"]
        + 0.005 * frm["normalized"]
    )

    if "scores" not in st.session_state:
        st.session_state.scores = []
    st.session_state.scores.append(total_score)

    response = f"Your translation quality was {score(total_score)}\n\n"

    acc_s = acc["score"]
    response += f"\nYour accuracy score is {score(acc_s)}:\n"

    for error in acc["errors"]:
        response += f" - {error['message']}\n"

    gre_s = gre["score"]
    ppl_s = ppl["score"]
    response += f"\nYour fluency score is {score(0.4 * gre_s + 0.6 * ppl_s)}\n"

    for error in gre["errors"]:
        response += f" - {error['message']}\n"

    for error in ppl["errors"]:
        response += f" - {error['message']}\n"

    lines = response.split("\n")
    for line in lines:
        for word in line.split():
            yield word + " "
            time.sleep(0.05)
        # After each line, yield a newline character or trigger a line break in Markdown
        yield "\n"


def translation_generator():
    # Check if translations are available and not empty
    if st.session_state.translations:
        # Randomly select a translation from the list
        st.session_state.german = random.choice(st.session_state.translations)["german"]
    else:
        st.error("No translations available.")
        return

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

if "translations" not in st.session_state:
    try:
        with open("translations.json", "r") as f:
            st.session_state.translations = json.loads(f.read())
            print(st.session_state.translations)
    except (FileNotFoundError, json.JSONDecodeError):
        st.session_state.translations = None
        # Create an empty translations dictionary if none exists
        st.error(
            "No previous translations found. Starting with an empty translation history."
        )

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
