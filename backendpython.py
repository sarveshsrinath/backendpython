from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.responses import JSONResponse
from datetime import datetime
import cv2
import numpy as np
import logging
from typing import List
#import requests

# Configure logging settings (optional)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Create a console handler and set the level to DEBUG
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# Create a formatter and add it to the handler
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)

# Add the console handler to the root logger
logging.getLogger().addHandler(console_handler)

app = FastAPI()

# Load pre-trained face detection model (you need to download this and adjust the path accordingly)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

@app.post("/detect-face/")
async def detect_face(unique_id: str= Form(...), file: UploadFile = File(...)):
    logging.info('In detect_face')

    if file.content_type.startswith('image/') is False:
        raise HTTPException(status_code=400, detail=f'File \'{file.filename}\' is not an image.')

    try:
        # Read image contents
        image_stream = await file.read()
        image = np.frombuffer(image_stream, np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_GRAYSCALE)

        image_dimensions = {"width": image.shape[1], "height": image.shape[0]}

        # Perform face detection
        faces = face_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30))

        print("No of faces :", len(faces))

        # Prepare the response
        if len(faces) == 0:
            return JSONResponse(content={"unique_id": unique_id, "face_detected": False})
        else:
            # Assuming the first face has the highest confidence score
            #confidence_score = faces[0][-1] if len(faces[0]) == 5 else 1  # OpenCV does not provide confidence score by default
            #return JSONResponse(content={"unique_id": unique_id, "face_detected": True, "confidence_score": confidence_score})
            face_coordinates = []
            for (x, y, w, h) in faces:
                face_coordinates.append({'x': int(x), 'y': int(y), 'width': int(w), 'height': int(h)})
            return JSONResponse(content={"unique_id": unique_id, "face_detected": True, "image_dimensions": image_dimensions,  "faces": face_coordinates})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/", response_class=HTMLResponse)
async def get_home():
    html_content = """
    <!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Camera App</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding:0;
            box-sizing: border-box;
        }
        header {
            text-align: center;
            padding: 20px;
            background-color: #f0f0f0;
            border-bottom: 2px solid #333;
        }
        .class-info{
            text-align: center;
            margin-bottom: 20px;
        }
        button {
            height: 3rem;
            border: 2px solid #333;
            background-color: #fff;
            color: #333;
            cursor: pointer;
        }
        button:hover{
            background-color: #333;
            color: #fff;
        }
        #video-tag,
        #image-tag{
            border: 2px solid #333;
        }
        div {
            border: 2px solid #ccc;
            padding: 10px;
            margin: 10px;
        }
        #take-photo-button {
            height: 8rem; 
            width: 12rem;
        }
        #imageContainer {
            float:left;
            position: relative;
            display: inline-block;
        }

        #overlay {
            position: absolute;
            top: 0;
            left: 0;
        }

    </style>
</head>
<body>
    <header>
        <h1>Delhi Public School Bangalore East</h1>
        <p class="class-info">Class 12I Project 2023-24</p>
    </header>

<div style="display:flex; justify-content:center"><button id="start" onclick="start()" style="height: 3rem;">start camera!</button>
    <button id="check" style="margin-left: 6rem;height:3rem;" onclick="stop()">stop camera!</button></div>

<div>
    <div style="display: flex;justify-content: center;height: auto;">
        <p><video id="video-tag" width="320" height="180" autoplay /></p>
    </div>
    <div id="imageContainer">
        <img id="image-tag" width="240"> </img>
        <canvas id="overlay" width="240"></canvas>
    </div>
</div>

<div style="display:flex;justify-content: center;height: auto;">
    <div>
        <button onclick="takePhoto()">take photo</button>
        <button id="clear-photo-button" onclick="clearPhoto()">clear photo</button>
        <button id="check-image-button" onclick="checkImage()">check face</button>
    </div>
    <div hidden>
        <label for="pan-slider">Pan</label>
        <input id="pan-slider" min="0" , max="0" name="pan" title="Pan" type="range" />
        <output id="pan-slider-value"></output>
    </div>
    <div hidden>
        <label for="tilt-slider">Tilt</label>
        <input id="tilt-slider" min="0" , max="0" name="tilt" title="Tilt" type="range" />
        <output id="tilt-slider-value"></output>
    </div>
    <div hidden>
        <label for="zoom-slider">Zoom</label>
        <input id="zoom-slider" min="0" , max="0" name="zoom" title="Zoom" type="range" />
        <output id="zoom-slider-value"></output>
    </div>
</div>
<input type="file" id="imageInput" style="display: none;">
<script>
    const constraints = {
        video: { width: { exact: 320 }, pan: true, tilt: true, zoom: true }
    };
    var videoTag = document.getElementById('video-tag');
    var imageTag = document.getElementById('image-tag');
    var panSlider = document.getElementById("pan-slider");
    var panSliderValue = document.getElementById("pan-slider-value");
    var tiltSlider = document.getElementById("tilt-slider");
    var tiltSliderValue = document.getElementById("tilt-slider-value");
    var zoomSlider = document.getElementById("zoom-slider");
    var zoomSliderValue = document.getElementById("zoom-slider-value");
    var tempImageSrc = imageTag.src;
    var imageCapturer;

    function start() {
        navigator.mediaDevices.getUserMedia(constraints)
            .then(gotMedia)
            .catch(e => { console.error('getUserMedia() failed: ', e); });
    }

    function gotMedia(mediastream) {
        console.log("Inside gotMedia! - Sarvesh is here!");
        videoTag.srcObject = mediastream;
        document.getElementById('start');

        var videoTrack = mediastream.getVideoTracks()[0];
        imageCapturer = new ImageCapture(videoTrack);

        // Timeout needed in Chrome, see https://crbug.com/711524
        setTimeout(() => {
            const capabilities = videoTrack.getCapabilities()
            const settings = videoTrack.getSettings();

            // Check whether pan is supported or not.
            if (capabilities.pan) {
                // Map pan to a slider element.
                panSlider.min = capabilities.pan.min;
                panSlider.max = capabilities.pan.max;
                panSlider.step = capabilities.pan.step;
                panSlider.value = settings.pan;
                panSlider.oninput = function (event) {
                    panSliderValue.value = panSlider.value;
                    videoTrack.applyConstraints({ advanced: [{ pan: event.target.value }] });
                };
                panSlider.parentElement.hidden = false;
            }

            // Check whether tilt is supported or not.
            if (capabilities.tilt) {
                // Map tilt to a slider element.
                tiltSlider.min = capabilities.tilt.min;
                tiltSlider.max = capabilities.tilt.max;
                tiltSlider.step = capabilities.tilt.step;
                tiltSlider.value = settings.tilt;
                tiltSlider.oninput = function (event) {
                    tiltSliderValue.value = tiltSlider.value;
                    videoTrack.applyConstraints({ advanced: [{ tilt: event.target.value }] });
                };
                tiltSlider.parentElement.hidden = false;
            }

            // Check whether zoom is supported or not.
            if (capabilities.zoom) {
                // Map zoom to a slider element.
                zoomSlider.min = capabilities.zoom.min;
                zoomSlider.max = capabilities.zoom.max;
                zoomSlider.step = capabilities.zoom.step;
                zoomSlider.value = settings.zoom;
                zoomSlider.oninput = function (event) {
                    zoomSliderValue.value = zoomSlider.value;
                    videoTrack.applyConstraints({ advanced: [{ zoom: event.target.value }] });
                };
                zoomSlider.parentElement.hidden = false;
            }
        }, 500);

    }

    function takePhoto() {

        imageCapturer.takePhoto()
            .then((blob) => {
                console.log("Photo taken: " + blob.type + ", " + blob.size + "B")
                imageTag.src = URL.createObjectURL(blob);
                const file = new File([blob], 'captured_photo.png', { type: 'image/png' });
                // Create a FileList containing the File object
                const fileList = new DataTransfer();
                fileList.items.add(file);

                // Set the FileList as the value of the imageInput
                imageInput.files = fileList.files;
            })
            .catch((err) => {
                console.error("takePhoto() failed: ", err);
            });
    }
    function clearPhoto() {
            imageTag.src=tempImageSrc;
            var overlay = document.getElementById('overlay');
            var ctx = overlay.getContext('2d');
            ctx.clearRect(0, 0, overlay.width, overlay.height);
            
        }
    function stop(){
        if (videoTag.srcObject) {
           const tracks = videoTag.srcObject.getTracks();
           tracks.forEach(track => track.stop());
           // Clear the video source
           videoTag.srcObject = null;
       }
  }
  function checkImage() {
    const input = document.getElementById('imageInput');

    if (input.files && input.files[0]) {
        const imageFile = input.files[0];
        const formData = new FormData();
        const timestamp = new Date().getTime();
        formData.append("unique_id", timestamp);
        formData.append("file", imageFile);

        fetch('/detect-face/', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            console.log(data);
            alert('Face detection result: ' + (data.face_detected ? 'Face Detected' : 'No Face Detected'));
            
            var canvas = document.getElementById('overlay');
            var ctx = canvas.getContext('2d');


            console.log('Image dimensions: width -' + data.image_dimensions.width + ' height -' + data.image_dimensions.height);
            data.faces.forEach(function (face) {
                console.log('Canvas width:' + canvas.width + 'Canvas height:' + canvas.height + 'Face x:' + face.x + 'Face y:' + face.y);
                console.log('Input width:' + input.files[0].width + 'Overlay width:' + overlay.width);
                var scaleFactor = data.image_dimensions.width / overlay.width;
                console.log('Scale Factor:' + scaleFactor);
                var scaledX = face.x / scaleFactor;
                var scaledY = face.y / scaleFactor;
                var scaledWidth = face.width / scaleFactor;
                var scaledHeight = face.height / scaleFactor;

                console.log('Canvas width:' + canvas.width + 'Canvas height:' + canvas.height + 'Scaled box x:' + scaledX + 'Scaled box y:' + scaledY);
                ctx.beginPath();
                //ctx.rect(face.x, face.y, face.width, face.height);
                ctx.rect(scaledX, scaledY, scaledWidth, scaledHeight);
                ctx.lineWidth = 2;
                ctx.strokeStyle = 'black';
                ctx.fillStyle = 'black';
                ctx.stroke();
            });

            // Update the original image with the canvas content
            //imageTag.src = canvas.toDataURL('image/png');
        })
        .catch(error => {
            console.error('Error:', error);
        });
    }
  }

</script>
</body>
</html>
"""
    return html_content

