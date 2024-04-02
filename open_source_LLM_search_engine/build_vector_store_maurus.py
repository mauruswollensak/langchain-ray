import os
import shutil
import tempfile
import time
from langchain.document_loaders import ReadTheDocsLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from embeddings import LocalHuggingFaceEmbeddings

FAISS_INDEX_PATH = "faiss_index"

# Determine the script's directory
working_dir = os.path.dirname(os.path.realpath(__file__))
# Locate the test files directory
test_files_dir = os.path.join(os.path.dirname(working_dir), "test_files")

print(f"Working directory: {working_dir}")
print(f"Test files directory: {test_files_dir}")

# Create a temporary directory for .txt files
temp_dir = tempfile.mkdtemp()

# Copy .txt files to the temporary directory
for file in os.listdir(test_files_dir):
    if file.endswith('.txt'):
        shutil.copy(os.path.join(test_files_dir, file), temp_dir)

# Initialize ReadTheDocsLoader with the temporary directory
loader = ReadTheDocsLoader(temp_dir)

# Set up the text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,  # Consider adjusting based on the average length of your documents
    chunk_overlap=20,
    length_function=len,
)

# Load documents and create chunks
print("Loading documents ...")
start_time = time.time()
docs = loader.load()
print(f"Number of documents loaded: {len(docs)}")
chunks = text_splitter.create_documents(
    [doc.page_content for doc in docs], metadatas=[doc.metadata for doc in docs]
)
print(f"Number of chunks created: {len(chunks)}")
elapsed_time = time.time() - start_time
print(f"Time taken to load and split documents: {elapsed_time} seconds.")

# Proceed if chunks were created
if chunks:
    # Embed the chunks
    print("Loading chunks into vector store ...")
    start_time = time.time()
    embeddings = LocalHuggingFaceEmbeddings("multi-qa-mpnet-base-dot-v1")
    db = FAISS.from_documents(chunks, embeddings)
    db.save_local(FAISS_INDEX_PATH)
    elapsed_time = time.time() - start_time
    print(f"Time taken to embed chunks and save: {elapsed_time} seconds.")
else:
    print("No chunks to embed. Process halted.")

# Clean up the temporary directory
shutil.rmtree(temp_dir)
