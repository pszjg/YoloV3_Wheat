# YoloV3_Wheat
An implementation of the YOLO framework for a custom dataset (Wheat) on a Windows 10 machine using Anaconda and Python

Thank you to https://github.com/experiencor for the initial implementation

Changes to support multiple networks, more optimisers, further data augmentation and more are being made in this version. Stay tuned for more

<b>Installation:</b>

There were numerous problems getting the installation to work on Windows 10, the configuration that is used here is:

1. Download Anaconda3 4.3.0.1
2. run 'conda install -c aaronzs tensorflow-gpu' on the anaconda command line
3. run 'pip install keras==2.1.5' on the anaconda command line
4. run 'pip install opencv-python' on the anaconda command line
5. downgrade protobuf to 3.6.0 using 'pip install protobuf==3.6.0'

The latest version of keras (2.2 at the time of writing) had numerous errors when compiling, as did the latest version of protobuf (3.6.1 at the time of writing) so both were downgraded, aaronzs tensorflow-gpu installs the necessary cudatoolkit, tensorboard and so on. It did install the latest version of protobuf which I downgraded in step 5
