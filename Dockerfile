FROM python:3.12-slim

WORKDIR /app

COPY requirements-dashboard.txt ./
RUN pip install --no-cache-dir -r requirements-dashboard.txt

COPY . .

EXPOSE 8501

CMD streamlit run dashboard.py --server.address=0.0.0.0 --server.port=${PORT:-8501}
