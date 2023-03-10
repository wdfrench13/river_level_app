FROM python:3.9-slim
COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt
COPY src/ ./
CMD gunicorn -b 0.0.0.0:8050 app:server
# ENTRYPOINT ['python', 'src/app.py']