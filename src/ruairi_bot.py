from dotenv import load_dotenv
from openai import OpenAI
import json
import chromadb
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pypdf import PdfReader
import gradio as gr
import utils
from pathlib import Path
import logging
import sys

# Set up logging
def setup_logging():
    """Configure logging for both development and production"""
    
    # Configure root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Remove existing handlers to avoid duplicates
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    
    # Set formatter
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    
    # Add handler to logger
    logger.addHandler(console_handler)
    
    # Only allow our application modules to log at INFO level
    app_modules = [
        'ruairi_bot',
        'utils',
        '__main__',
        'root'  # for direct logger.info() calls
    ]
    
    for module_name in app_modules:
        logging.getLogger(module_name).setLevel(logging.INFO)
    
    # TODO: is there a neater way to do this, e.g. set all to WARNING then override?
    # Or possibly use the pyproject.toml
    # Explicitly set common third-party modules to WARNING level
    # This is more reliable than trying to set all modules at once
    third_party_modules = [
        'httpx',
        'gradio', 
        'gradio_client',
        'uvicorn',
        'fastapi',
        'chromadb',
        'openai',
        'langchain',
        'urllib3',
        'requests'
    ]
    
    for module_name in third_party_modules:
        logging.getLogger(module_name).setLevel(logging.WARNING)
    
    return logger

# Initialize logging
logger = setup_logging()

# database and enviroment setup
load_dotenv(override=True)

# Environment variable validation
required_env_vars = {
    'OPENAI_API_KEY': 'OpenAI API key for embedding and chat completions'
}

try:
    openai = OpenAI()
    logger.info("OpenAI client initialized successfully")
except Exception as e:
    logger.critical(f"Failed to initialize OpenAI client: {e}")
    logger.error("This usually means your OPENAI_API_KEY is invalid or there's a network issue")
    raise SystemExit("Cannot start without OpenAI client")

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
try:
    chroma_client = chromadb.PersistentClient(path=str(CHROMA_STORE_PATH))
    logger.info(f"ChromaDB initialized at: {CHROMA_STORE_PATH}")
except Exception as e:
    logger.critical(f"Failed to initialize ChromaDB: {e}")
    logger.error(f"Check that the directory {CHROMA_STORE_PATH} is accessible")
    raise SystemExit("Cannot start without database")

# init Q&A chroma collection
try:
    interview_collection = chroma_client.get_or_create_collection("interview_qna")

    if not QUESTIONS_DB_PATH.exists():
        logger.critical(f"Questions database not found at: {QUESTIONS_DB_PATH}")
        raise SystemExit("Cannot start without questions database")
    
    file_hash = utils.get_file_hash(QUESTIONS_DB_PATH)
    stored_hash = utils.get_stored_file_hash(interview_collection, doc_type="interview_qna")

    if file_hash == stored_hash:
        logger.info("Interview Q&A already embedded and unchanged.")
    else:
        logger.info("Interview Q&A changed or missing — re-embedding...")

        # Delete existing non-hash entries (leave the stored hash doc alone)
        try:
            existing_ids = interview_collection.get()["ids"]
            real_ids = [id for id in existing_ids if not id.startswith("__file_hash__")]
            if real_ids:
                interview_collection.delete(ids=real_ids)
        except Exception as e:
            logger.error(f"Failed to clean old embeddings: {e}")
                
        # Load Q&A data from JSON
        try:
            with open(QUESTIONS_DB_PATH, "r") as f:
                qna_data = json.load(f)
            logger.info(f"Loaded {len(qna_data)} Q&A pairs")
        except json.JSONDecodeError as e:
            logger.critical(f"Invalid JSON in questions database: {e}")
            raise SystemExit("Questions database is corrupted")
        except Exception as e:
            logger.critical(f"Failed to read questions database: {e}")
            raise SystemExit("Cannot access questions database")

        # Add Q&A documents to a chroma collection
        try:
            for i, item in enumerate(qna_data):
                    
                question = item['question']
                answer = item['answer']
                
                try:
                    question_embedding = openai.embeddings.create(
                        input=[question],
                        model="text-embedding-3-small"
                    ).data[0].embedding
                except Exception as e:
                    logger.error(f"Failed to embed question {i}: {e}")
                    continue

                interview_collection.add(
                    documents=[question],  # List of 1 string
                    embeddings=[question_embedding],  # List of 1 embedding
                    metadatas=[{"answer": answer}],  # List of 1 dict
                    ids=[f"qna-{i}"]  # List of 1 ID
                )
            logger.info(f"Successfully embedded {len(qna_data)} Q&A pairs")
        except Exception as e:
            logger.critical(f"Failed to embed Q&A data: {e}")
            raise SystemExit("Failed to setup interview collection")

        # Store updated hash
        try:
            utils.store_file_hash_in_chroma(interview_collection, file_hash, doc_type='interview_qna')
        except Exception as e:
            logger.error(f"Failed to store file hash: {e}")

except Exception as e:
    logger.critical(f"Failed to setup interview collection: {e}")
    raise SystemExit("Cannot start without interview data")

# Init thesis chroma collection
try:
    thesis_collection = chroma_client.get_or_create_collection("thesis_chunks")
    
    if not THESIS_PATH.exists():
        logger.error(f"Thesis not found at: {THESIS_PATH}")
        logger.error("Thesis-related functionality will be disabled")
        thesis_collection = None
    else:
        file_hash = utils.get_file_hash(THESIS_PATH)
        stored_hash = utils.get_stored_file_hash(thesis_collection, doc_type="thesis")

        if file_hash == stored_hash:
            logger.info("Thesis already embedded and unchanged.")
        else:
            logger.info("Thesis changed or missing — re-embedding...")

            # Delete existing non-hash entries (leave the stored hash doc alone)
            try:
                existing_ids = thesis_collection.get()["ids"]
                real_ids = [id for id in existing_ids if not id.startswith("__file_hash__")]
                if real_ids:
                    thesis_collection.delete(ids=real_ids)
            except Exception as e:
                logger.error(f"Failed to clean old thesis embeddings: {e}")

            # Load in thesis text 
            try:
                thesis_text = utils.extract_text_from_pdf(THESIS_PATH)
                logger.info(f"Extracted thesis text ({len(thesis_text)} characters)")
            except Exception as e:
                logger.critical(f"Failed to extract text from thesis PDF: {e}")
                logger.error("Check that the thesis PDF is valid and readable")
                raise SystemExit("Cannot process thesis PDF")

            # Chunk thesis
            try:
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=500,
                    chunk_overlap=50,
                    separators=["\n\n", "\n", ".", " ", ""]
                )
                chunks = splitter.split_text(thesis_text)
                logger.info(f"Split thesis into {len(chunks)} chunks")
            except Exception as e:
                logger.critical(f"Failed to chunk thesis text: {e}")
                raise SystemExit("Cannot process thesis text")

            # Embed and store in thesis collection
            try:
                successful_embeddings = 0
                for i, chunk in enumerate(chunks):
                    try:
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
                        successful_embeddings += 1
                    except Exception as e:
                        logger.error(f"Failed to embed thesis chunk {i}: {e}")
                        continue
                        
                logger.info(f"Successfully embedded {successful_embeddings}/{len(chunks)} thesis chunks")
                
                if successful_embeddings == 0:
                    logger.critical("Failed to embed any thesis chunks")
                    raise SystemExit("No thesis data available")
                    
            except Exception as e:
                logger.critical(f"Failed to embed thesis data: {e}")
                raise SystemExit("Failed to setup thesis collection")

            # Store updated hash
            try:
                utils.store_file_hash_in_chroma(thesis_collection, file_hash, doc_type='thesis')
            except Exception as e:
                logger.error(f"Failed to store thesis file hash: {e}")

except Exception as e:
    logger.critical(f"Failed to setup thesis collection: {e}")
    raise SystemExit("Cannot start without thesis processing")

# Load in linkedin text
try:
    if not LINKEDIN_PATH.exists():
        logger.error(f"LinkedIn PDF not found at: {LINKEDIN_PATH}")
        linkedin = "LinkedIn profile information not available."
    else:
        reader = PdfReader(LINKEDIN_PATH)
        linkedin = ""
        for page in reader.pages:
            text = page.extract_text()
            if text:
                linkedin += text
        
        if not linkedin.strip():
            logger.error("LinkedIn PDF appears to be empty or unreadable")
            linkedin = "LinkedIn profile information not available."
        else:
            logger.info(f"Loaded LinkedIn profile ({len(linkedin)} characters)")
            
except Exception as e:
    logger.error(f"Failed to read LinkedIn PDF: {e}")
    linkedin = "LinkedIn profile information not available."

# Load in the pre-written summary
try:
    if not SUMMARY_PATH.exists():
        logger.critical(f"Summary file not found at: {SUMMARY_PATH}")
        raise SystemExit("Summary file is required")
    
    with open(SUMMARY_PATH, "r", encoding="utf-8") as f:
        summary = f.read()
        
    if not summary.strip():
        logger.critical("Summary file is empty")
        raise SystemExit("Summary file cannot be empty")
        
    logger.info(f"Loaded summary ({len(summary)} characters)")
    
except Exception as e:
    logger.critical(f"Failed to read summary file: {e}")
    raise SystemExit("Cannot start without summary")

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
    try:
        # TODO: does gradio already do this?
        # Input validation
        if not message or not message.strip():
            return "❌ **Error**: Please provide a message to chat with me!"
        
        # Check if there is any relevant info in any of the collection in chroma
        try:
            retrieved_info = utils.retrieve_rag_context(openai, message, chroma_client)
            rag_context = f"\n\n## Retrieved Info:\n{retrieved_info}" if retrieved_info else ""
        except Exception as e:
            logger.error(f"RAG retrieval failed: {e}")
            # Continue without RAG context
            rag_context = ""
            # Show warning in chat but continue
            # We'll add this as a prefix to the final response

        # Add the rag context to the prompt, will add nothing in case there was no suitable context found
        full_prompt = system_prompt + rag_context

        messages = [{"role": "system", "content": full_prompt}] + history + [{"role": "user", "content": message}]

        done = False
        response = None
        max_iterations = 5  # Prevent infinite loops
        iteration_count = 0

        while not done and iteration_count < max_iterations:
            iteration_count += 1
            
            try:
                # Call the LLM with the above prompt
                response = openai.chat.completions.create(model="gpt-4o-mini", messages=messages, tools=tools)
            except Exception as e:
                # API failure - be loud about it
                logger.error(f"OpenAI API call failed: {e}")
                error_msg = f"🚨 **OpenAI API Error**: {str(e)}\n\n"
                error_msg += "This could be due to:\n"
                error_msg += "• API key issues\n"
                error_msg += "• Rate limiting\n" 
                error_msg += "• Network connectivity\n"
                error_msg += "• OpenAI service outage\n\n"
                error_msg += "Please try again in a moment. If the problem persists, check your network connection."
                logger.error(f"OpenAI API call failed: {e}")
                return error_msg

            try:
                # Check if the LLM wants to call a tool
                finish_reason = response.choices[0].finish_reason
                 
                if finish_reason == "tool_calls":
                    # route to the correct tool and run it. 
                    message_obj = response.choices[0].message
                    tool_calls = message_obj.tool_calls
                    
                    try:
                        results = utils.handle_tool_calls(tool_calls)
                        messages.append(message_obj)
                        messages.extend(results)
                    except Exception as e:
                        logger.error(f"Tool call failed: {e}")
                        # Return error to user but continue
                        error_msg = "⚠️ **Tool Error**: Failed to execute tool function.\n"
                        error_msg += f"Error: {str(e)}\n\n"
                        error_msg += "I can still chat with you, but some functionality may not work properly."
                        return error_msg
                else:
                    done = True
                    
            except (KeyError, IndexError, AttributeError) as e:
                logger.error(f"Unexpected response format from OpenAI: {e}")
                error_msg = "🚨 **Response Processing Error**: Received unexpected response format from AI service.\n"
                error_msg += f"Technical details: {str(e)}\n\n"
                error_msg += "Please try rephrasing your question or try again."
                return error_msg

        if iteration_count >= max_iterations:
            logger.warning("Hit maximum tool call iterations")
            return "⚠️ **System Error**: The conversation got stuck in a loop. Please start a new conversation."

        if not response:
            logger.error("No response generated")
            return "🚨 **System Error**: Failed to generate a response. Please try again."

        try:
            final_response = response.choices[0].message.content
            if not final_response:
                return "⚠️ **Warning**: I generated an empty response. Please try rephrasing your question."
                
            # Add RAG warning if retrieval failed but we continued
            if rag_context == "" and "RAG retrieval failed" in str(locals().get('e', '')):
                final_response = "⚠️ *Note: Some background information may be unavailable due to a database issue.*\n\n" + final_response
                
            return final_response
            
        except (KeyError, IndexError, AttributeError) as e:
            logger.error(f"Failed to extract response content: {e}")
            return f"🚨 **Response Extraction Error**: {str(e)}\n\nPlease try again."
            
    except Exception as e:
        # Catch-all for any other unexpected errors
        logger.error(f"Unexpected error in chat function: {e}")
        error_msg = "🚨 **Unexpected System Error**: Something went wrong.\n"
        error_msg += f"Error details: {str(e)}\n\n"
        error_msg += "Please try again or refresh the page if the problem persists."
        return error_msg

# Launch the Gradio interface
try:
    logger.info("Starting Gradio interface...")
    gr.ChatInterface(chat, type="messages").launch()
except Exception as e:
    logger.critical(f"Failed to launch Gradio interface: {e}")
    raise SystemExit("Cannot start web interface")
