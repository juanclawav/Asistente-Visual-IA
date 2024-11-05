from fastapi import FastAPI, UploadFile, Form
from ai_assistant.rags import multimodal_rag_query
from ai_assistant.config import Config

app = FastAPI(title="AI Vision")

@app.post("/query/")
async def query_image(query: str = Form(...), image: UploadFile = Form(...)):
    # Save the uploaded image
    image_path = Config.INPUT_IMAGE_PATH / image.filename
    with open(image_path, "wb") as buffer:
        buffer.write(await image.read())
        
    # Perform the multimodal query
    response = multimodal_rag_query(query, image_path)
    return {"response": response}