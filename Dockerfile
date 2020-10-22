FROM ubuntu:20.04
MAINTAINER Diego Molla <dmollaaliod@gmail.com>

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get -y install nano python3-numpy python3-scipy python3-matplotlib python3-pip emacs tmux

RUN pip3 install -U nltk
RUN pip3 install -U scikit-learn

ADD ROUGE-1.5.5.tgz /root/rouge/
RUN perl -MCPAN -e 'install XML::DOM'
RUN mkdir /root/rouge/models
RUN mkdir /root/rouge/summaries

RUN pip3 install -U tensorflow

ADD *.py /root/code/
ADD nnmodels/*.py /root/code/nnmodels/
ADD summariser/*.py /root/code/summariser/

RUN python3 /root/code/dockersetup.py

RUN mkdir /root/code/crossvalidation
RUN mkdir /root/code/savedmodels

RUN pip3 install -U rouge progressbar2 transformers

WORKDIR /root/code
