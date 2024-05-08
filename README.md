# Handwave Recognition
This project recognizes hand wave movements in videos using the YoloV8 model, which is trained on an augmented HaGRID dataset. The system includes custom scripts for movement recognition and is designed for deployment to an AWS SageMaker Endpoint.

### Demonstration Video

![Watch the video example](examples/output_example.gif)

## Training the Model
The model utilizes [Ultralytics YoloV8](https://github.com/ultralytics/ultralytics), originally trained on the [Coco dataset](https://docs.ultralytics.com/datasets/detect/coco/#dataset-structure). Since the Coco dataset lacks 'palm' classes necessary for hand wave recognition, this project employs a portion of the [HaGRID dataset](https://github.com/hukenovs/hagrid). Image and bounding box augmentation was performed using the Albumentations library to enhance training outcomes. The training was executed in a Google Colab environment using a subset of the HaGRID dataset due to computational constraints.

Before running the provided Jupyter Notebook `yolov8.ipynb`, ensure all dependencies are installed by running `pip install -r requirements.txt` in your environment. This will install necessary packages including Ultralytics YoloV8, Albumentations, and other supporting libraries.

[Link to download trained weights on Google Drive (50Mb)](https://drive.google.com/file/d/1Ocl76s68G9AbiE1FHc01CnDkITYq3yzh/view?usp=sharing) 

## Model Inference
The `handwave_recognition.py` script enables local model inferencing. Utilizing Nvidia CUDA technology for acceleration, this script can perform inference in real-time. Modify the input source in `cv2.VideoCapture("your input")` to change the video source. Setting this to `0` will use the web camera, allowing real-time operation. Ensure your environment is set up with all required libraries by installing them from the `requirements.txt` file.

## AWS SageMaker Endpoint Inference
Deployment code for an AWS SageMaker endpoint is located in the `endpoint` directory. This endpoint relies on a Docker image hosting a server, as outlined in the `endpoint/inference.py` script. A Dockerfile is provided for building this image.

### Intended Workflow:
1. **Video Upload**: Users upload a video to the "input" folder in an AWS S3 Bucket.
2. **Lambda Trigger**: Uploading a video triggers an AWS Lambda function, executing the code in `endpoint/aws_lambda_code.py`.
3. **Endpoint Invocation**: The Lambda function invokes the SageMaker endpoint with the video path.
4. **Video Processing**: `endpoint/inference.py` downloads the video, processes it, and uploads the results with annotated bounding boxes and recognized waves to the "output" folder in the same S3 Bucket.

The Docker infrastructure and the Lambda function work together to streamline the process from video upload to generating processed outputs, demonstrating how effectively this solution scales and operates in real-world scenarios.

