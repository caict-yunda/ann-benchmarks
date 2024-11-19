FROM ann-benchmarks


RUN echo "151.101.108.223 files.pythonhosted.org" >> /etc/hosts
RUN export http_proxy=http://172.20.3.88:1088
RUN export https_proxy=http://172.20.3.88:1088
RUN pip3 install dingosdk
RUN pip3 install dingodb==0.0.18rc5
RUN python3 -c 'from dingodb import SDKVectorDingoDB, SDKClient'
