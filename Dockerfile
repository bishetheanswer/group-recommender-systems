FROM python:3.11-slim-bookworm

RUN apt update && apt install -y gcc

WORKDIR /code

COPY requirements.txt /code/

RUN pip install -r requirements.txt

COPY . /code/

CMD ["/bin/bash"]
