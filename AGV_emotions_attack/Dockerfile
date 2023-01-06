FROM tensorflow/tensorflow:latest-gpu

# update the systems
RUN apt-get update -y && apt-get upgrade -y
RUN apt-get install wget -y

# things for print and display keras models as imgs
RUN apt-get install graphviz -y
RUN pip install pydot pydotplus graphviz

# math and charts libs
RUN pip install matplotlib pandas seaborn 

# sklearn lib. Offers a LOT OF THINGS. Most used for splitting dataset into train set and test/validation set
RUN pip install sklearn

# opencv dependecies
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6 -y

WORKDIR /
COPY AGV-Project/AGV_emotions_attack/requirements_gpu.txt requirements_gpu.txt

RUN pip install -r requirements_gpu.txt

RUN pip3 install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html

RUN apt-get install python3-tk -y

RUN pip install grad-cam