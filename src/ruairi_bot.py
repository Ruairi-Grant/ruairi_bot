from dotenv import load_dotenv
from openai import OpenAI
import json
import chromadb
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pypdf import PdfReader
import gradio as gr
import utils
from pathlib import Path

# database and enviroment setup
load_dotenv(override=True)

openai = OpenAI()

name = "Ruairi Grant"

# Path configuration - use relative paths from project root
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "me"
CHROMA_STORE_PATH = PROJECT_ROOT / "chroma_store"

# File paths
QUESTIONS_DB_PATH = DATA_DIR / "questions_db.json"
THESIS_PATH = DATA_DIR / "Thesis.pdf"
LINKEDIN_PATH = DATA_DIR / "linkedin.pdf"
SUMMARY_PATH = DATA_DIR / "summary.txt"

# VectorDB setup
chroma_client = chromadb.PersistentClient(path=str(CHROMA_STORE_PATH))

# init Q&A chroma collection
interview_collection = chroma_client.get_or_create_collection("interview_qna")

file_hash = utils.get_file_hash(QUESTIONS_DB_PATH)
stored_hash = utils.get_stored_file_hash(interview_collection, doc_type="interview_qna")

if file_hash == stored_hash:
    print("Interview Q&A already embedded and unchanged.")
else:
    print("Interview Q&A changed or missing — re-embedding.")

    # Delete existing non-hash entries (leave the stored hash doc alone)
    existing_ids = interview_collection.get()["ids"]
    real_ids = [id for id in existing_ids if not id.startswith("__file_hash__")]
    if real_ids:
        interview_collection.delete(ids=real_ids)
            
    # Load Q&A data from JSON
    with open(QUESTIONS_DB_PATH, "r") as f:
        qna_data = json.load(f)

    # Add Q&A documents to a chroma collection
    for i, item in enumerate(qna_data):
        question = item['question']
        answer = item['answer']
        question_embedding = openai.embeddings.create(
            input=[question],
            model="text-embedding-3-small"
        ).data[0].embedding

        interview_collection.add(
            documents=[question],  # List of 1 string
            embeddings=[question_embedding],  # List of 1 embedding
            metadatas=[{"answer": answer}],  # List of 1 dict
            ids=[f"qna-{i}"]  # List of 1 ID
        )

    # Step 4: Store updated hash
    utils.store_file_hash_in_chroma(interview_collection, file_hash, doc_type='interview_qna')

# Init theisis chroma collection
thesis_collection = chroma_client.get_or_create_collection("thesis_chunks")

file_hash = utils.get_file_hash(THESIS_PATH)
stored_hash = utils.get_stored_file_hash(thesis_collection, doc_type="thesis")

if file_hash == stored_hash:
    print("Thesis already embedded and unchanged.")
else:
    print("thesis changed or missing — re-embedding.")

    # Delete existing non-hash entries (leave the stored hash doc alone)
    existing_ids = thesis_collection.get()["ids"]
    real_ids = [id for id in existing_ids if not id.startswith("__file_hash__")]
    if real_ids:
        thesis_collection.delete(ids=real_ids)

    # Load in thesis text 
    thesis_text = utils.extract_text_from_pdf(THESIS_PATH)

    # Chunk thesis
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = splitter.split_text(thesis_text)

    # Embed and store in thesis collection
    for i, chunk in enumerate(chunks):
        embedding = openai.embeddings.create(
            input=[chunk],
            model="text-embedding-3-small"
        ).data[0].embedding

        thesis_collection.add(
            documents=[chunk],
            embeddings=[embedding],
            metadatas=[{"chunk_index": i}],
            ids=[f"thesis-{i}"]
        )

    # Step 4: Store updated hash
    utils.store_file_hash_in_chroma(thesis_collection, file_hash, doc_type='thesis')

# Load in linkedin text
reader = PdfReader(LINKEDIN_PATH)
linkedin = ""
for page in reader.pages:
    text = page.extract_text()
    if text:
        linkedin += text

# Load in the pre-written summary
with open(SUMMARY_PATH, "r", encoding="utf-8") as f:
    summary = f.read()

# Load in tool decriptions to provide this functionality to the model
tools = [{"type": "function", "function": utils.record_user_details_json},
        {"type": "function", "function": utils.record_unknown_question_json}]

# Build the system prompt
system_prompt = f"You are acting as {name}. You are answering questions on {name}'s website, \
particularly questions related to {name}'s career, background, skills and experience. \
Your responsibility is to represent {name} for interactions on the website as faithfully as possible. \
You are given a summary of {name}'s background and LinkedIn profile which you can use to answer questions. \
You also may be provided with relevant info about {name}'s thesis and answers to standard interview questions. \
Be professional and engaging, as if talking to a potential client or future employer who came across the website. \
If you don't know the answer to any question, use your record_unknown_question tool to record the question that you couldn't answer, even if it's about something trivial or unrelated to career. \
If the user is engaging in discussion, try to steer them towards getting in touch via email; ask for their email and record it using your record_user_details tool. "

system_prompt += f"\n\n## Summary:\n{summary}\n\n## LinkedIn Profile:\n{linkedin}\n\n"
system_prompt += f"With this context, please chat with the user, always staying in character as {name}."

# Chat callback function to orgainize the workflow to respond to a chat message
def chat(message, history):
    # Check if there is any relevant info in any of the collection in chroma
    retrieved_info = utils.retrieve_rag_context(openai, message, chroma_client)
    rag_context = f"\n\n## Retrieved Info:\n{retrieved_info}" if retrieved_info else ""

    # Add the rag context to the prompt, will add nothing in case there was no suitable context found
    full_prompt = system_prompt + rag_context

    messages = [{"role": "system", "content": full_prompt}] + history + [{"role": "user", "content": message}]

    done=False

    while not done:
        try:
            # Call the LLM with the above prompt
            response = openai.chat.completions.create(model="gpt-4o-mini", messages=messages, tools=tools)
        except Exception as e:
            raise RuntimeError("Failed to get completion") from e

        # Check if the LLM wants to call a tool
        finish_reason = response.choices[0].finish_reason
         
        if finish_reason=="tool_calls":
            # route to the correct tool and run it. 
            message = response.choices[0].message
            tool_calls = message.tool_calls
            results = utils.handle_tool_calls(tool_calls)
            messages.append(message)
            messages.extend(results)
        else:
            done = True

    response_text = response.choices[0].message.content
        
    # Debug output
    debug_output = (
        f"\n\n--- PROMPT START ---\n \
        {full_prompt} \
        \n--- PROMPT END ---\n\n \
        {response_text}"
    )
    # return debug_output
    return response.choices[0].message.content

gr.ChatInterface(chat, type="messages").launch()
