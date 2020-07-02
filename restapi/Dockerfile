FROM python:3.7

WORKDIR /casestudy

COPY api.py .
COPY requirements.txt .
COPY predictions.csv.zip .

RUN unzip predictions.csv.zip predictions.csv
RUN pip install -r requirements.txt

EXPOSE 8080

CMD [ "python", "/casestudy/api.py", "/casestudy/predictions.csv" ]
