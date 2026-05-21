# Claude Code Instructions for api/

- imports must use: import database (NOT from api import database)
- imports must use: from models import ... (NOT from api.models import ...)
- uvicorn runs from inside api/ directory, so relative imports are correct
