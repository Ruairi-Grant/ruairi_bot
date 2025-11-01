todo

## High Priority (Deployment Blockers)
- **CRITICAL**: Remove unused debug code (lines 170-176 in ruairi_bot.py)


## Medium Priority (Code Quality)
- change the routing, just include chunks from both
- fix issues with thesis by adding thesis specific questions to the questions_db
- add rate limiting
- Split ruairi_bot.py into separate modules (embedding, chat, database)
- Add type hints and docstrings to functions
- Add input validation and sanitization for user messages
- Implement proper ChromaDB connection management
- Clean up unused dependencies in pyproject.toml
- tidy up error handling as much as possible

## Deployment Setup
- Create Dockerfile with proper base image
- Add health check endpoints for monitoring
- Set up production server (replace Gradio dev server)
- Add environment-specific configurations
- Create deployment documentation/README

## Low Priority
- move to multi agent model - possibly to explore improving rag searching
- add guardrails
- Add comprehensive test suite
- Implement connection pooling for better performance
- Add monitoring and metrics collection
- add introduction message and nice title
- properly connect the hugginface space with the github repo
- move backend elsewhere from huggingface so i can have better data file management
