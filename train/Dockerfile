FROM python
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
COPY train.py train.py
CMD ["python", "train.py"]
