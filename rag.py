from vertexai.preview import rag
from vertexai.preview.generative_models import GenerativeModel, Tool
import vertexai

# Create a RAG Corpus, Import Files, and Generate a response

# TODO(developer): Update and un-comment below lines
# project_id = "PROJECT_ID"
# display_name = "test_corpus"
# paths = ["https://drive.google.com/file/123", "gs://my_bucket/my_files_dir"]  # Supports Google Cloud Storage and Google Drive Links

# Initialize Vertex AI API once per session
vertexai.init(project=project_id, location="us-central1")

# Create RagCorpus
rag_corpus = rag.create_corpus(display_name=display_name)

# Import Files to the RagCorpus
response = rag.import_files(
    rag_corpus.name,
    paths,
    chunk_size=512,  # Optional
    chunk_overlap=100,  # Optional
)

# Direct context retrieval
response = rag.retrieval_query(
    rag_corpora=[rag_corpus.name],
    text="What is RAG and why it is helpful?",
    similarity_top_k=10,
)
print(response)

# Enhance generation
# Create a RAG retrieval tool
rag_retrieval_tool = Tool.from_retrieval(
    retrieval=rag.Retrieval(
        source=rag.VertexRagStore(
            rag_corpora=[rag_corpus.name],  # Currently only 1 corpus is allowed.
            similarity_top_k=3,  # Optional
        ),
    )
)
# Create a gemini-pro model instance
rag_model = GenerativeModel(
    model_name="gemini-1.0-pro-002", tools=[rag_retrieval_tool]
)

# Generate response
response = rag_model.generate_content("What is RAG and why it is helpful?")
print(response.text)