from llama_index.core import PromptTemplate

# Template for question-answering in RAG setup
qa_tmpl_str = (
    "Context information is below.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Given the context information and not prior knowledge, "
    "answer the query based on the image provided.\n"
    "Query: {query_str}\n"
    "Answer: "
)

qa_template = PromptTemplate(qa_tmpl_str)