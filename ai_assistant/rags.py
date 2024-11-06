
from llama_index.core.query_engine import SimpleMultiModalQueryEngine
from llama_index.multi_modal_llms.openai import OpenAIMultiModal
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import SimpleDirectoryReader, StorageContext
from ai_assistant.prompts import qa_template
from llama_index.core.indices import MultiModalVectorStoreIndex
from ai_assistant.config import Config
from qdrant_client.http.exceptions import UnexpectedResponse
import qdrant_client

# Set up OpenAI LLM
openai_mm_llm = OpenAIMultiModal(model="gpt-4o", api_key=Config.OPENAI_API_KEY)


# Initialize Qdrant vector stores for text and image data
client = qdrant_client.QdrantClient(path="qdrant_mm_db")

def collection_exists(client, collection_name):
    try:
        client.get_collection(collection_name)
        return True
    except UnexpectedResponse:
        return False

# Check if collections already exist in Qdrant
text_collection_exists = collection_exists(client, "text_collection")
# Initialize Qdrant vector stores for text and image data
if text_collection_exists:
    print("Reusing existing text collection in Qdrant.")
else:
    print("Creating a new text collection in Qdrant.")


# Create or reuse the vector stores
text_store = QdrantVectorStore(client=client, collection_name="text_collection")

storage_context = StorageContext.from_defaults(vector_store=text_store)

# Only build the index if collections do not already exist
if not text_collection_exists:
    documents = SimpleDirectoryReader(str(Config.DATA_PATH)).load_data()
    index = MultiModalVectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
    )
else:
    # Load existing index without rebuilding
    index = MultiModalVectorStoreIndex(storage_context=storage_context)

# Query Engine
query_engine = index.as_query_engine(
    llm=openai_mm_llm, text_qa_template=qa_template
)


def multimodal_rag_query(query_str, image_path):
    if image_path:
        image_description = openai_mm_llm.complete(
            prompt="Generate a detailed text description for these images, if there are clothes focus on the clothing in detail, if the image is a poster or document or invitation, focus on the details of the information contained in it.",
            image_documents=SimpleDirectoryReader(image_path).load_data(),
        )
        response = query_engine.query(f"{query_str}\n\nImage description: {image_description}")
    else :
        response = query_engine.query(f"{query_str}")

    
    return response