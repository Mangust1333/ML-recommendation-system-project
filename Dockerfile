FROM python:3.11

WORKDIR /work

ENV PIP_CACHE_DIR=/root/.cache/pip

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

RUN pip install --no-cache-dir jupyterlab notebook ipykernel

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=${JUPYTER_TOKEN}"]
