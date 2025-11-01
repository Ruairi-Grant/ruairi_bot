from pypdf import PdfReader
import json
import hashlib
import logging
from config import get_config

# Get logger for this module
logger = logging.getLogger(__name__)

# Get configuration
cfg = get_config()

# TODO: record this somewhere persistent
def record_user_details(email, name="Name not provided", notes="not provided"):
    try:
        logger.info(f"Recording interest from {name} with email {email} and notes {notes}")
        # Here you could add actual database/file storage logic
        return {"recorded": "ok"}
    except Exception as e:
        logger.error(f"Failed to record user details: {e}")
        return {"error": str(e)}

# TODO: record this somewhere persistent
def record_unknown_question(question):
    try:
        logger.info(f"Recording unknown question: {question}")
        # Here you could add actual logging/storage logic
        return {"recorded": "ok"}
    except Exception as e:
        logger.error(f"Failed to record unknown question: {e}")
        return {"error": str(e)}

def is_thesis_question(message: str) -> bool:
    thesis_keywords = [
        "thesis", "research", "paper", "experiment", "model", "simulation",
        "hypothesis", "methodology", "dataset", "findings", "conclusion"
    ]
    message_lower = message.lower()
    return any(keyword in message_lower for keyword in thesis_keywords)

record_user_details_json = {
    "name": "record_user_details",
    "description": "Use this tool to record that a user is interested in being in touch and provided an email address",
    "parameters": {
        "type": "object",
        "properties": {
            "email": {
                "type": "string",
                "description": "The email address of this user"
            },
            "name": {
                "type": "string",
                "description": "The user's name, if they provided it"
            }
            ,
            "notes": {
                "type": "string",
                "description": "Any additional information about the conversation that's worth recording to give context"
            }
        },
        "required": ["email"],
        "additionalProperties": False
    }
}

record_unknown_question_json = {
    "name": "record_unknown_question",
    "description": "Always use this tool to record any question that couldn't be answered as you didn't know the answer",
    "parameters": {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "The question that couldn't be answered"
            },
        },
        "required": ["question"],
        "additionalProperties": False
    }
}

def get_file_hash(filepath):
    with open(filepath, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

def store_file_hash_in_chroma(collection, file_hash, doc_type):
    collection.upsert(
        ids=[f"__file_hash__::{doc_type}"],
        documents=["file hash record"],
        metadatas=[{"file_hash": file_hash}],
        embeddings=[[0.0]*1536]  # dummy embedding (must match model size)
    )

def get_stored_file_hash(collection, doc_type):
    try:
        result = collection.get(ids=[f"__file_hash__::{doc_type}"])
        return result["metadatas"][0]["file_hash"]
    except (IndexError, KeyError):
        return None

def handle_tool_calls(tool_calls):
    results = []
    for tool_call in tool_calls:
        tool_name = tool_call.function.name
        arguments = json.loads(tool_call.function.arguments)
        logger.debug(f"Tool called: {tool_name}")
        tool = globals().get(tool_name)
        result = tool(**arguments) if tool else {}
        results.append({"role": "tool","content": json.dumps(result),"tool_call_id": tool_call.id})
    return results


def retrieve_from_qna(openai, user_message, database, top_k=None, threshold=None):
    if top_k is None:
        top_k = cfg.qna_n_results
    if threshold is None:
        threshold = cfg.qna_threshold
        
    try:
        embedding = openai.embeddings.create(
            input=[user_message],
            model="text-embedding-3-small"
        ).data[0].embedding

        result = database.query(query_embeddings=[embedding], n_results=top_k)

        logger.debug("Q&A Search - Distances: %s", result['distances'])
        logger.debug("Q&A Search - Questions: %s", result['documents'])

        if result['distances'][0][0] < threshold:
            return result['metadatas'][0][0]['answer']
        return ""  # No match found
    except Exception as e:
        logger.error(f"Q&A retrieval failed: {e}")
        return ""
    
def retrieve_from_thesis(openai, user_message, database, top_k=None, threshold=None):
    if top_k is None:
        top_k = cfg.thesis_n_results
    if threshold is None:
        threshold = cfg.thesis_threshold
        
    try:
        embedding = openai.embeddings.create(
            input=[user_message],
            model="text-embedding-3-small"
        ).data[0].embedding

        result = database.query(query_embeddings=[embedding], n_results=top_k)

        logger.debug("Thesis Search - Distances: %s", result['distances'])
        logger.debug("Thesis Search - Found chunks: %d", len(result['documents'][0]))

        retrieved_chunks = []
        for doc, score in zip(result['documents'][0], result['distances'][0]):
            if score < threshold:
                retrieved_chunks.append(doc)

        return "\n\n".join(retrieved_chunks)
    except Exception as e:
        logger.error(f"Thesis retrieval failed: {e}")
        return ""


def retrieve_rag_context(openai, user_message, database, top_k=2):
    if is_thesis_question(user_message):
        logger.debug("Routing to thesis_chunks collection")
        return retrieve_from_thesis(openai, user_message, database.get_collection("thesis_chunks"), top_k=top_k)
    else:
        logger.debug("Routing to qna collection")
        return retrieve_from_qna(openai, user_message, database.get_collection("interview_qna"),  top_k=top_k)
    
def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text
    return text