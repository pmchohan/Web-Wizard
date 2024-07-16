import numpy as np
import streamlit as st
from bs4 import BeautifulSoup
import os, pickle, requests, faiss
from langchain.text_splitter import RecursiveCharacterTextSplitter
from gensim.models import FastText
from dotenv import load_dotenv
import google.generativeai as genai

os.environ['KMP_DUPLICATE_LIB_OK']='True'
load_dotenv()
genai.configure(api_key=os.environ['GEMINI_API_KEY'])
model = genai.GenerativeModel('gemini-pro')
# models/gemini-1.0-pro
# models/gemini-1.0-pro-001
# models/gemini-1.0-pro-latest

# after gemini init
class unstructured:
    def __init__(self, text, md):
        self.page_content = text
        self.metadata = {'source': md}

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
}
def extract_text(url):
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, 'html.parser')
    text = soup.get_text()
    ud = unstructured(text.strip().lower(), url)
    return ud

st.title("Web Wizard 🧙‍♂️")
st.sidebar.title("Article URLs (Blog, Documentation, NEWS etc.)")
urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    if url:
        urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")

for _ in range(17):
    st.sidebar.text('')

# Now add your names
st.sidebar.markdown('<p style="font-size: medium; color: lightgray; text-align: right; margin-bottom: 0;">Abdullah Faisal</p>', unsafe_allow_html=True)
st.sidebar.markdown('<p style="font-size: medium; color: lightgray; text-align: right; margin-bottom: 0;">Rehan Hanif</p>', unsafe_allow_html=True)
st.sidebar.markdown('<p style="font-size: medium; color: lightgray; text-align: right; margin-bottom: 0;">Ali Adil Waseem</p>', unsafe_allow_html=True)
st.sidebar.markdown('<p style="font-size: medium; color: lightgray; text-align: right; margin-bottom: 0;">Ali Manan</p>', unsafe_allow_html=True)

file_path = "faiss_store.pkl"
chunks_path = "chunks.pkl"
embedder_path = "embedder.pkl"

main_placeholder = st.empty()

if process_url_clicked:
    # load data
    global chunks
    main_placeholder.text("Data Loading...Started...✅✅✅")
    texts = [extract_text(url) for url in urls]
    
    # split data
    main_placeholder.text("Breaking bones of data...✅✅✅")
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=300
    )
    docs = text_splitter.split_documents(texts)
    chunks = [doc.page_content for doc in docs]
    # create embeddings and save it to FAISS index
    main_placeholder.text("Creating Embeddings...✅✅✅")
    ft = FastText(chunks, min_count=10, vector_size=200)
    embeddings = [ft.wv[chunk] for chunk in chunks]
    embeddings = np.array(embeddings)

    # create a FAISS index
    main_placeholder.text("Creating Vector DB...✅✅✅")
    embedding_dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(embedding_dim)

    # add embeddings to the index
    index.add(embeddings)

    # Save the FAISS index to a pickle file
    with open(file_path, "wb") as f:
        pickle.dump(index, f)
    with open(chunks_path, "wb") as f:
        pickle.dump(chunks, f)
    with open(embedder_path, "wb") as f:
        pickle.dump(ft, f)

query = main_placeholder.text_input("Question: ")
if query:
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            index = pickle.load(f)
        with open(chunks_path, "rb") as f:
            chunks = pickle.load(f)
        with open(embedder_path, "rb") as f:
            ft = pickle.load(f)
        
        vec = ft.wv[query]
        svec = np.array(vec).reshape(1, -1)
        _, I = index.search(svec, k=6)
        row_indices = I.tolist()[0]
        most_similar = [chunks[i].strip().lower() for i in row_indices]
        context = " ".join(most_similar)
        question = query+"  Don't reply to things that are out of context. find good and balanced answer from context below don't add anything on your own, you can clean up the answer a bit: \n```{}```".format(context)
        response = model.generate_content(question)
        st.header("Answer")
        st.write(response.text)
