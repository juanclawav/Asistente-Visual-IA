from llama_index.core.tools import FunctionTool, ToolMetadata
from ai_assistant.rags import multimodal_rag_query
from ai_assistant.rags import OpenAIMultiModal
from ai_assistant.config import Config
import qdrant_client

# Tool to generate detailed descriptions of images
def describe_image(image_path: str) -> str:
    """
    Generate a description of an image based on its content.
    
    Parameters:
    - image_path: Path to the image file.

    Returns:
    - Description of the image as a string.
    """
    # Call multimodal_rag_query with an empty query string for image description
    return multimodal_rag_query("", image_path)


# Tool to perform multimodal query (image + text)
def multimodal_query(query_str: str, image_path: str) -> str:
    """
    Generate a response based on a query and image content.

    Parameters:
    - query_str: User query.
    - image_path: Path to the image file.

    Returns:
    - Response from the multimodal RAG model.
    """
    return multimodal_rag_query(query_str, image_path)


# Define FunctionTool instances for these tools
image_description_tool = FunctionTool.from_defaults(
    fn=describe_image,
    metadata=ToolMetadata(
        name="describe_image",
        description="Generates a detailed description of an image."
    ),
    return_direct=False
)

multimodal_query_tool = FunctionTool.from_defaults(
    fn=multimodal_query,
    metadata=ToolMetadata(
        name="multimodal_query",
        description="Answers a query based on provided text and image content."
    ),
    return_direct=False
)