# DOCKER FILE FOR DEPLOYMENT OF BDO GPT USER ACCEPTANCE TESTING PHASE
FROM python:3.9-slim

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8501 

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "Main   .py", "--server.port=8501", "--server.address=0.0.0.0"]