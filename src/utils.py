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

        # Always return the best match, but indicate if it's below threshold
        if result['distances'][0]:
            distance = result['distances'][0][0]
            similarity = 1 - distance  # Convert distance to similarity
            answer = result['metadatas'][0][0]['answer']
            
            # Check if it meets the threshold
            if distance < threshold:
                return answer, similarity, True  # True = meets threshold
            else:
                # Return anyway but mark as below threshold
                return answer, similarity, False  # False = below threshold
        
        return "", None, False  # No results at all
    except Exception as e:
        logger.error(f"Q&A retrieval failed: {e}")
        return "", None, False
    
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
        chunk_scores = []
        chunks_below_threshold = []
        scores_below_threshold = []
        
        for doc, score in zip(result['documents'][0], result['distances'][0]):
            similarity = 1 - score
            if score < threshold:
                # Above threshold - include these
                retrieved_chunks.append(doc)
                chunk_scores.append(similarity)
            else:
                # Below threshold - keep track separately
                chunks_below_threshold.append(doc)
                scores_below_threshold.append(similarity)

        # If we have good matches, return those
        if retrieved_chunks:
            return "\n\n".join(retrieved_chunks), chunk_scores, True
        
        # If no good matches, return the best available matches anyway
        elif chunks_below_threshold:
            return "\n\n".join(chunks_below_threshold), scores_below_threshold, False
            
        return "", [], False
    except Exception as e:
        logger.error(f"Thesis retrieval failed: {e}")
        return "", [], False


def retrieve_rag_context(openai, user_message, database, top_k=2):
    logger.debug("Retrieving from both Q&A and thesis collections")
    
    # Get results from both collections
    qna_result, qna_score, qna_meets_threshold = retrieve_from_qna(openai, user_message, database.get_collection("interview_qna"), top_k=top_k)
    thesis_result, thesis_scores, thesis_meets_threshold = retrieve_from_thesis(openai, user_message, database.get_collection("thesis_chunks"), top_k=top_k)
    
    # Build prompt content (only include results that meet threshold)
    prompt_results = []
    
    if qna_result and qna_meets_threshold:
        prompt_results.append(f"## Interview Q&A:\n{qna_result}")
    
    if thesis_result and thesis_meets_threshold:
        prompt_results.append(f"## Thesis Information:\n{thesis_result}")
    
    prompt_text = "\n\n".join(prompt_results) if prompt_results else ""
    
    # Build debug content (include everything with scores and warnings)
    debug_results = []
    
    if qna_result:
        if qna_score is not None:
            threshold_marker = "" if qna_meets_threshold else " ⚠️ BELOW THRESHOLD"
            qna_header = f"## Interview Q&A (similarity: {qna_score:.3f}{threshold_marker}):\n{qna_result}"
        else:
            qna_header = f"## Interview Q&A:\n{qna_result}"
        debug_results.append(qna_header)
        
    if thesis_result:
        if thesis_scores:
            avg_score = sum(thesis_scores) / len(thesis_scores)
            threshold_marker = "" if thesis_meets_threshold else " ⚠️ BELOW THRESHOLD"
            thesis_header = f"## Thesis Information (avg similarity: {avg_score:.3f}, {len(thesis_scores)} chunks{threshold_marker}):\n{thesis_result}"
        else:
            thesis_header = f"## Thesis Information:\n{thesis_result}"
        debug_results.append(thesis_header)
    
    debug_text = "\n\n".join(debug_results) if debug_results else ""
    
    # Return structured data
    return {
        "prompt_content": prompt_text,  # Clean content for AI model (only good matches)
        "debug_content": debug_text,    # Full content with scores for debugging
        "metadata": {
            "qna_similarity": qna_score,
            "qna_meets_threshold": qna_meets_threshold,
            "thesis_similarities": thesis_scores,
            "thesis_meets_threshold": thesis_meets_threshold
        }
    }
    
def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text
    return text