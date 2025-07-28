import fitz  # PyMuPDF for PDF handling
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter

# Load and extract text from PDF (first 5 pages by default)
def load_pdf(file_path, max_pages=5):
    doc = fitz.open(file_path)
    text = ""
    for page_num in range(min(max_pages, len(doc))):
        page = doc.load_page(page_num)
        text += page.get_text()
    return text

text = load_pdf("load.pdf")             # Use PDF file

# Preview first 1000 characters
print("🔍 First 1000 characters of content:\n")
print(text[:1000])

# Chunking the text

# 1. Fixed-size Chunking
fixed_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
fixed_chunks = fixed_splitter.split_text(text)
print(f"\n📦 Total fixed-size chunks: {len(fixed_chunks)}")
print("📄 First fixed chunk:\n")
print(fixed_chunks[0])

# 2. Recursive Chunking (more context-aware)
recursive_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
)
recursive_chunks = recursive_splitter.split_text(text)
print(f"\n📦 Total recursive chunks: {len(recursive_chunks)}")
print("📄 First recursive chunk:\n")
print(recursive_chunks[0])