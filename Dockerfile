FROM nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04

WORKDIR /app

# Dependências do sistema
RUN apt -y update && \
    apt -y install python3 python3-pip libsndfile1 build-essential cmake nodejs npm && \
    apt -y clean && rm -rf /var/lib/apt/lists/*

# Copia requirements
COPY requirements.txt .

# Atualiza pip e instala dependências
RUN pip3 install --upgrade pip && \
    pip3 install -r requirements.txt && \
    pip3 cache purge && \
    rm requirements.txt

# Instala Jupyter Lab
RUN pip3 install jupyterlab

EXPOSE 8888
EXPOSE 6006

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--allow-root", "--no-browser"]
