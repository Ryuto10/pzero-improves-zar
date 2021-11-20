FROM ubuntu:20.04
USER root

# install
RUN apt-get update && apt-get install -y \
    wget \
    curl \
    tar \
    bzip2 \
    vim \
    make \
    gcc \
    g++ \
    python3.8 \
    python3-pip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir /root/tmp

# download & unzip
WORKDIR /root/tmp
RUN wget "https://drive.google.com/uc?export=download&id=0B4y35FiV1wh7QVR6VXJ5dWExSTQ" -O CRF++-0.58.tar.gz \
    && wget "https://drive.google.com/uc?export=download&id=0B4y35FiV1wh7cENtOXlicTFaRUE" -O mecab-0.996.tar.gz \
    && curl -sc cookie "https://drive.google.com/uc?export=download&id=0B4y35FiV1wh7SDd1Q1dUQkZQaUU" > /dev/null \
    && curl -Lb cookie "https://drive.google.com/uc?export=download&confirm=$(awk '/_warning_/ {print $NF}' cookie)&id=0B4y35FiV1wh7SDd1Q1dUQkZQaUU" -o cabocha-0.69.tar.bz2 \
    && curl -sc cookie "https://drive.google.com/uc?export=download&id=0B4y35FiV1wh7X2pESGlLREpxdXM" > /dev/null \
    && curl -Lb cookie "https://drive.google.com/uc?export=download&confirm=$(awk '/_warning_/ {print $NF}' cookie)&id=0B4y35FiV1wh7X2pESGlLREpxdXM" -o mecab-jumandic-7.0-20130310.tar.gz \
    && tar -xf CRF++-0.58.tar.gz \
    && tar -xf mecab-0.996.tar.gz \
    && tar -xf mecab-jumandic-7.0-20130310.tar.gz \
    && tar -xf cabocha-0.69.tar.bz2

# complie
WORKDIR /root/tmp/CRF++-0.58
RUN ./configure \
    && make \
    && make install

WORKDIR /root/tmp/mecab-0.996
RUN ./configure --with-charset=utf8 \
    && make \
    && make install \
    && echo '/usr/local/lib' >> /etc/ld.so.conf.d/local.conf \
    && ldconfig

WORKDIR /root/tmp/mecab-jumandic-7.0-20130310
RUN ./configure --with-charset=utf8 \
    && make \
    && make install

WORKDIR /root/tmp/cabocha-0.69
RUN ./configure --with-charset=UTF8 \
    && make \
    && make install \
    && ldconfig

# set path for python
RUN rm -rf /root/tmp \
    && ln -s /usr/bin/python3.8 /usr/bin/python \
    && rm -rf /usr/bin/python3 \
    && ln -s /usr/bin/python3.8 /usr/bin/python3

# copy
COPY docker/mecabrc /usr/local/etc/mecabrc
COPY poetry.lock .
COPY pyproject.toml .

# pip install
RUN pip3 install --upgrade pip \
    && pip3 install --no-cache-dir poetry
    # && poetry config virtualenvs.create false \
    # && poetry install \
    # && poetry cache clear pypi --all -n \
    # && rm -rf /root/.cache/pip

WORKDIR /code
