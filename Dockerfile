FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY proxy.py .
COPY AppleRootCA-G3.cer .

EXPOSE 8080

CMD ["python", "proxy.py"]
