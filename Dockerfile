FROM tensorflow/tensorflow:2.18.0-gpu-jupyter
WORKDIR /app
RUN echo 2
COPY requirements.txt .
RUN apt -y update && apt -y install libsndfile1 build-essential cmake && apt -y clean
RUN pip install -r requirements.txt && pip cache purge
RUN rm requirements.txt
EXPOSE 8888
EXPOSE 6006

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--allow-root", "--no-browser"]