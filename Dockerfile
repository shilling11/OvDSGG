FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04

# Set system environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_BREAK_SYSTEM_PACKAGES=1

ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
ENV FORCE_CUDA=1

ENV TORCH_CUDA_ARCH_LIST="7.0;8.0;8.6;9.0"

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    git \
    ninja-build \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    wget \
    ffmpeg \
    tzdata \
    && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3 /usr/bin/python

RUN python3 -m pip install --upgrade --ignore-installed pip setuptools wheel

RUN pip install torch torchvision

RUN pip install git+https://github.com/openai/CLIP.git

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

RUN cd models/ops && \
    rm -rf build/ dist/ *.egg-info && \
    sed -i "s/if torch.cuda.is_available() and CUDA_HOME is not None:/if (torch.cuda.is_available() or os.getenv('FORCE_CUDA')) and CUDA_HOME is not None:/" setup.py && \
    # patch all occurrences of .type() with .scalar_type() in cuda file
    sed -i 's/\.type()/.scalar_type()/g' src/cuda/ms_deform_attn_cuda.cu && \
    # patch all occurrences of .type().is_cuda() with .is_cuda()
    sed -i 's/\.scalar_type().is_cuda()/.is_cuda()/g' src/cuda/ms_deform_attn_cuda.cu && \
    # patch header file
    sed -i 's/\.type().is_cuda()/.is_cuda()/g' src/ms_deform_attn.h && \
    sed -i 's/\.type()/.scalar_type()/g' src/ms_deform_attn.h && \
    TORCH_CUDA_ARCH_LIST="7.0;8.0;8.6;9.0" FORCE_CUDA=1 python setup.py build

RUN cp models/ops/build/lib.*/MultiScaleDeformableAttention*.so models/ops/

CMD ["/bin/bash"]