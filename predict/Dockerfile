FROM python
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
COPY predict.py predict.py
EXPOSE 8080
CMD ["python", "predict.py"]

