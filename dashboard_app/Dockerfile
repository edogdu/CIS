FROM python:3.11-bookworm

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .
EXPOSE 8050

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8050"]