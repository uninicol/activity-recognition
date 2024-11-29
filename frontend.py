import os
from ultralytics import YOLO
import streamlit as st
import torch
from langchain.agents import initialize_agent
from langchain.agents.agent import  AgentType

from langchain.llms import Ollama
from langchain.memory import ConversationBufferMemory
from loguru import logger

from tools import ObjectDetectionTool, PromptGeneratorTool


class App:
    def __init__(self, device) -> None:

        # if "agent" not in st.session_state:
        llm = Ollama(model="llama3.2:3b")
        self._agent = initialize_agent(
            agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
            tools=[
                ObjectDetectionTool().setup(
                    YOLO(
                        "C:/Users/Informatica_UNICAM/Desktop/activity-recognition/"
                        "runs/detect/custom_yolo7/weights/best.pt"
                    )
                ),
                PromptGeneratorTool().setup(llm),
            ],
            llm=llm,
            memory=ConversationBufferMemory(return_messages=True),
            verbose=True,
            max_iterations=3,
        )
        st.session_state["agent"] = self._agent

    def _upload_image(self) -> None:
        uploaded_image = st.file_uploader("Upload an image")
        if uploaded_image:
            tmp_dir = "tmp/"
            os.makedirs(tmp_dir, exist_ok=True)
            temp_file_path = os.path.join(tmp_dir, uploaded_image.name)
            with open(temp_file_path, "wb") as file:
                file.write(uploaded_image.getvalue())
            st.sidebar.image(temp_file_path, width=200)
            self._process_image(temp_file_path)

    def _process_image(self, image_path: str) -> None:
        try:
            result = self._agent(
                f"Describe the image: {image_path} and detect objects with the description."
            )
            logger.debug(result)
        except Exception as e:
            logger.error(e)

    def run(self) -> None:
        llm = Ollama(model="llama3.2:3b")
        st.set_page_config(page_title="LLaMA Chat", layout="wide")
        st.title("Chat with LLaMA")
        with st.form("chat_form"):
            user_input = st.text_area("Enter your message:", height=100)
            submit_button = st.form_submit_button(label="Send")
        if submit_button and user_input.strip():
            with st.spinner("LLaMA is processing..."):
                # Invia l'input a LLaMA e ottieni la risposta
                try:
                    response = llm(user_input)  # Invio della richiesta
                    st.success("Response received!")
                    st.text_area("LLaMA's Response:", response, height=200, disabled=True)
                except Exception as e:
                    st.error(f"An error occurred: {e}")
        st.title("Image Auto Annotation")
        self._upload_image()
        st.sidebar.markdown("---")
        st.sidebar.slider(
            "Image quality", min_value=0, max_value=100, value=70, key="output_quality"
        )


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    App(device=device).run()
