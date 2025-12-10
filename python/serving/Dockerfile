# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# This is used to build a Docker image that includes the necessary dependencies
# for running HeAR as a microservice.

FROM tensorflow/serving:2.18.0

COPY ./python/data_processing /data_processing
COPY ./python/serving /serving
COPY ./LICENSE /LICENSE
RUN chmod a+x /serving/entrypoint.sh

# Copy model from build context to container
# Before building, copy the model to project root:
#   mkdir -p models && cp -r /home/ubuntu/shared/models/hear ./models/hear
# Then build: docker build -f python/serving/Dockerfile -t hear-serving .
# Or use build arg: docker build -f python/serving/Dockerfile --build-arg MODEL_SRC=./models/hear -t hear-serving .
ARG MODEL_SRC=./models/hear
COPY ${MODEL_SRC} /models/hear

# Install Miniconda with Python 3.12 (for TensorFlow 2.20.0 compatibility)
# Note: TensorFlow 2.20.0 and tensorflow-io-gcs-filesystem require Python <3.13
# Set non-interactive mode to avoid prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Shanghai
RUN apt-get update && \
    apt-get -y upgrade && \
    apt-get install -y --no-install-recommends tzdata && \
    ln -fs /usr/share/zoneinfo/Asia/Shanghai /etc/localtime && \
    dpkg-reconfigure --frontend noninteractive tzdata && \
    apt-get install -y git wget bzip2 build-essential cmake ninja-build && \
    wget https://repo.anaconda.com/miniconda/Miniconda3-py312_25.9.1-3-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh && \
    /opt/conda/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    /opt/conda/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r && \
    /opt/conda/bin/conda install -y python=3.12 && \
    /opt/conda/bin/conda clean -ya && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set conda to PATH
ENV PATH=/opt/conda/bin:$PATH
ENV PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple
ENV PIP_TRUSTED_HOST=mirrors.aliyun.com

# Install dependencies using python3 (default Python from conda)
RUN python3 -m venv /server-env && \
    /server-env/bin/python3 -m pip install --upgrade pip wheel setuptools && \
    /server-env/bin/python3 -m pip install \
        -r /serving/serving_framework/pip-install.txt && \
    /server-env/bin/python3 -m pip install \
        -r /serving/requirements.txt

# TODO(b/379159076) - Remove gcloud
# Install gcloud CLI
RUN apt-get install -y apt-transport-https curl gnupg

# Clone python-certifi to meet MPL 2.0 License source code mirroring terms
# RUN git clone https://github.com/certifi/python-certifi.git

# Clone python-frozendict to meet LGPL 3.0 License source code mirroring terms
# RUN git clone https://github.com/Marco-Sulla/python-frozendict.git

ENTRYPOINT ["/serving/entrypoint.sh"]