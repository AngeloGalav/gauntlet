# gauntlet

<p align="center">
  <img src="res//image.png" alt="Data Thanos" width="400px"/>
</p>

> _"I used the AI to destroy the AI"_
\- Thanos if he was a data scientist

Pending

## How to use:
Place the respective datasets folders in the data folder:
- CIFAKE contents into CIFAKE folder
- RealVsAiArt contents into RealVsAiArt folder

## TODOs
- [ ] Leftover models stuff
    - [ ] test standard CNN
    - [ ] Implement data augmentation in Mixed model (maybe turn data augmentation dataset creation step into a function?)

- [ ] do Xai stuff
    - [ ] try CAPTUM? (https://captum.ai/)
    - [ ] Combining Grad-CAM with Guided Backpropagation (?)
        - tried it, failed miserably (needs grad to be computed or smth)
    - [ ] batch lime/gradcam (no display, only save) (i.e. run and save all gradcam for a batch or collection of batches to see them later)
    - [x] test to see if we can use less layers in gradcam for speeding up the computation (result: layer4 should be fine for resnet).
    - [x] create gradcam/lime function that takes a SINGLE image as input (in the format found in server.py) and returns the plt object!!!
    - [ ] extend webapp_gradcam function to work with lime as well

- [ ] webapp frontend
    - [ ] dynamic model selection
    - [x] fix position of images/buttons (make them "stay put") (optional)
    - [x] add code for upload to webapp server
    - [x] fix image sizes
    - [ ] add radio buttons for lime/gradcam
    - [ ] add input for server address
    - [x] hide incoming image from server w/ loading animation until server response
    - [ ] error handling
        - [ ] handle server status
    - [ ] add a "resolution" slider (selects layers of gradcam)
    - [ ] (VERY OPTIONAL) display server console output on frontend
    - [ ] server url stuff
    - [ ] working server status

- [ ] webapp server
    - [x] create tester app
    - [x] create gradcam func for single image
    - [ ] select model stuff
    - [ ] select backend stuff

- [ ] Misc
    - [ ] Add text for various explanations
    - [ ] solve all the other code TODOs (ctrl+shift+f TODO)


## Data
https://www.kaggle.com/datasets/cashbowman/ai-generated-images-vs-real-images?resource=download

## Useful links
https://coderzcolumn.com/tutorials/artificial-intelligence/lime-explain-keras-image-classification-network-predictions

https://www.kaggle.com/code/hamedaraab/classification-of-real-and-ai-generated-images#Utilities