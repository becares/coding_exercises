import streamlit as st
import requests
from matplotlib import pyplot as plt
import json

# Streamed response emulator
def response_handler(prompt):
    input_data = {"scenario": prompt}
    success = requests.post(url="http://127.0.0.1:8000/predict", data=json.dumps(input_data))
    if success:
        return plt.imread("diagram.png")

st.title("PlantUML Scenario generator")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

with st.chat_message("assistant"):
    response = st.write("Hello! Welcome to the PlantUML scenario generator. \n \
                        To start creating diagrams, describe in the text box below your desired \
                        scenario and the Fine-Tuned Microsoft Phi 1.5 will generate the \
                        PlantUML code for you.")

# Display chat messages from history on app rerun
#for message in st.session_state.messages:
#    with st.chat_message(message["role"]):
#        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Type in here your scenario..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        response = st.write("Sure! Here you have your diagram:")
        img = st.image(response_handler(prompt))
        
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.session_state.messages.append({"role": "assistant", "content": img})

