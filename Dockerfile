FROM pytorch/pytorch:2.1.0-cuda12.1-runtime-ubuntu22.04

# Install Python dependencies
RUN pip install --no-cache-dir \
    transformers>=4.35.0 \
    accelerate>=0.24.0 \
    peft>=0.6.0 \
    datasets>=2.14.0 \
    trl>=0.7.0 \
    pyyaml \
    jupyter \
    jupyterlab

# Copy frozen-layer modules package
COPY frozen_layer_modules /frozen_layer_modules

# Copy training utilities
COPY training_utils.py /workspace/training_utils.py

# Copy Jupyter kernel startup script
COPY jupyter_startup.py /etc/jupyter/kernel_startup.py

# Make frozen_layer_modules importable everywhere
ENV PYTHONPATH=/frozen_layer_modules:/workspace:$PYTHONPATH

# Default to distillation OFF
ENV DISTILLATION=off

# Wire the startup script into every Jupyter kernel
RUN mkdir -p /root/.jupyter && \
    printf 'c.IPKernelApp.exec_files = ["/etc/jupyter/kernel_startup.py"]\n' \
    > /root/.jupyter/jupyter_notebook_config.py

EXPOSE 8888

WORKDIR /workspace

CMD ["jupyter", "lab", \
     "--ip=0.0.0.0", \
     "--no-browser", \
     "--allow-root", \
     "--NotebookApp.token=''", \
     "--NotebookApp.password=''"]
