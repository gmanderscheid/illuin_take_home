FROM ubuntu

RUN apt-get update && \
    apt-get install -y python3-pip 
    
WORKDIR /home

COPY . .

RUN pip install -r requirements.txt  

RUN python3 -m nltk.downloader stopwords
