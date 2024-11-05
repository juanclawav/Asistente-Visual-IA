import gradio as gr
from ai_assistant.rags import multimodal_rag_query
from pathlib import Path
from PIL import Image

import gtts

def chatbot_interface(query, image_path, language):
    if image_path:
            # Open the image using the path provided by Gradio
        image = Image.open(image_path)
    
    # Define the path to save the image temporarily
        temp_image_dir = Path("./ai_assistant/data/input_images")
        temp_image_path = Path("./ai_assistant/data/input_images/temp_image.jpeg")
    
    # Save the image
        image.save(temp_image_path, format="JPEG")
    

    # Perform the multimodal RAG query with the provided query and saved image path
    response = multimodal_rag_query(query, image_path=str(temp_image_dir) if image_path else None)
    tts = gtts.gTTS(str(response), lang=language)
    audio_path = "./response_audio.mp3"
    tts.save(audio_path)
    
    return str(response), audio_path

# Updated Gradio interface
iface = gr.Interface(
    fn=chatbot_interface,
    inputs=[gr.Textbox(label="Query"), gr.Image(type="filepath", label="Upload Image"),
            gr.Radio(choices=["en", "es"], label="Select Language", value="en") ],  # Set type to "filepath"
    
    outputs=[gr.Textbox(label="Response"), gr.Audio(label="Response Audio")],
    title="Multimodal AI Assistant"
)

if __name__ == "__main__":
    iface.launch()