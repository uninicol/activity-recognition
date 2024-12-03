import os
import sys
import time
from typing import Optional
from ultralytics import YOLO

from langchain.llms import Ollama
from langchain.schema.messages import HumanMessage
from langchain.tools import BaseTool
from loguru import logger
from PIL import Image

logger.remove()
logger.add(sys.stderr, level="INFO")

class PromptGeneratorTool(BaseTool):
    name: str = "Image object detection prompt generator tool"
    description: str = "Generate prompts based on the description of the image for object detection."

    llm: Optional[Ollama] = None

    def setup(self, llm: Ollama) -> "PromptGeneratorTool":
        self.llm = llm
        return self

    def _run(self, image_desc: str) -> str:
        logger.debug(f"Image description: {image_desc}")
        input_msg = [
            HumanMessage(
                content=(
                    "Remove stop words and useless words, only keep the 'objects', "
                    f"from the following sentence:\n\n{image_desc}\n\n"
                    "List the objects, separating each with a comma."
                )
            )
        ]
        gen_prompt = self.llm.generate(messages=input_msg).generations[0].text
        logger.debug(f"Generated prompt: {gen_prompt}")
        return gen_prompt

    def _arun(self, query: str):
        raise NotImplementedError


class ObjectDetectionTool(BaseTool):
    name: str = "Object detection on image tool"
    description: str = (
        "Perform object detection on an image (read an image path) with a text prompt."
    )

    model: Optional[YOLO] = None

    def setup(self, _model: YOLO) -> "ObjectDetectionTool":
        self.model = _model
        return self

    def _run(self, image_path, prompt: str) -> str:
        logger.debug(f"Image path: {image_path}, prompt: {prompt}")

        if not self.model:
            raise ValueError("The model has not been set up. Please call `setup` first.")

        result = self.model(image_path["title"])
        output_dir = "output/"
        os.makedirs(output_dir, exist_ok=True)
        now_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        output_img_path = os.path.join(output_dir, f"{now_time}_obj_detection.png")
        result[0].save(filename = output_img_path)

        return Image.open(output_img_path)

    def _arun(self, query: str):
        raise NotImplementedError

