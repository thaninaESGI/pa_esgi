import streamlit as st
from help_desk import HelpDesk
from dotenv import load_dotenv
load_dotenv()

# Cache the model to avoid reloading it every time
@st.cache_resource
def get_model():
    model = HelpDesk(new_db=True)
    return model

model = get_model()

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Comment puis-je vous aider ?"}]

# Display chat messages
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Input prompt from user
if prompt := st.chat_input("Comment puis-je vous aider ?"):
    # Add user's message to session state
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # Get answer from model
    result, sources = model.retrieval_qa_inference(prompt)

    # Format the response
    response = result + '  \n  \n' + sources

    # Add assistant's response to session state
    st.chat_message("assistant").write(response)
    st.session_state["messages"].append({"role": "assistant", "content": response})
