FROM python:3.6-onbuild

RUN pip install -r requirements.txt


COPY preprocessing.py /src/preprocessing.py
COPY config.py /src/config.py
COPY requirements.txt /src/requirements.txt

#Run the command
CMD ["python", "./preprocessing.py"]