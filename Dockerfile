FROM python:3.9-slim-bullseye

WORKDIR .

RUN apt-get -y update
RUN apt-get -y install graphviz graphviz-dev
RUN apt-get -y install zip unzip
RUN apt-get install build-essential -y
RUN apt-get -y install git

RUN pip install "matplotlib>=3.7.0,<3.8.0"
RUN pip install "pandas>=1.4.0,<1.6.0"
RUN pip install "dash>=2.6.0,<2.8.0"
RUN pip install "plotly>=5.14.1,<5.15.0"
RUN pip install "gensim>=4.3.0,<4.4.0"
RUN pip install "networkx>=3.0,<3.2"
RUN pip install "numpy>=1.22.0"
RUN pip install "community==1.0.0b1"
RUN pip install "python-louvain>=0.16"
RUN pip install "cltk>=1.1.0,<1.2.0"
RUN pip install "gunicorn>=20.1.0,<20.2.0"

COPY src src
COPY cltk-fetch.py cltk-fetch.py 

RUN python cltk-fetch.py

EXPOSE 8080
CMD cd src && gunicorn -b 0.0.0.0:8080 main:server
