FROM tensorflow/tensorflow:2.9.2-gpu-jupyter
MAINTAINER Joseph Ouyang
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub
RUN apt-get update
RUN apt-get install -y ffmpeg libsm6 libxext6 npm
RUN npm install -g configurable-http-proxy
RUN pip install --upgrade pip
RUN pip install jupyterhub
RUN pip install tensorflow_datasets scikit-image opencv-python pandas jupyterlab jupyterhub-nativeauthenticator jupyterlab-nvdashboard tensorflow-addons
RUN pip install -q git+https://github.com/tensorflow/examples.git
RUN pip install centrosome
EXPOSE 8888
ENTRYPOINT ["jupyter", "lab","--ip=0.0.0.0","--allow-root"]