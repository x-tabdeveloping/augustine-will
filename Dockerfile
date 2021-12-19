FROM python:3.9-slim-bullseye

WORKDIR .

RUN apt-get -y update
RUN apt-get -y install graphviz graphviz-dev
RUN apt-get -y install zip unzip
RUN apt-get install build-essential -y
RUN apt-get -y install git

RUN pip install matplotlib pandas dash plotly gensim networkx numpy community python-louvain cltk

COPY . .

EXPOSE 8080
CMD ["python", "src/main.py"]