# config.py
import os

# API Keys
VOYAGE_API_KEY = "pa-juUgtBE09xGKeOTEuAYwXVjsjPhXwYK9tPGTVuStLqA"
ANTHROPIC_API_KEY = "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"

# Embedding model - voyage-code-3 is best for programming documentation
VOYAGE_MODEL = "voyage-code-3"

# Claude model
CLAUDE_MODEL = "claude-3-7-sonnet-20250219"

# Vector DB settings
VECTOR_DB_PATH = "./vector_db"

# Document processing settings
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200