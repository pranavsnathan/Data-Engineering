FROM python:3.6-onbuild

RUN pip install -r requirements.txt


COPY static /src/static
COPY templates /src/templates
COPY app.py /src/app.py
COPY requirements.txt /src/requirements.txt
COPY RFFinalModel.pkl /src/RFFinalModel.pkl

EXPOSE 5000

ENTRYPOINT ["python"]

#Run the command
CMD ["app.py"]