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
    - [x] batch lime/gradcam (no display, only save) (i.e. run and save all gradcam for a batch or collection of batches to see them later)
    - [x] test to see if we can use less layers in gradcam for speeding up the computation (result: layer4 should be fine for resnet).
    - [x] create gradcam/lime function that takes a SINGLE image as input (in the format found in server.py) and returns the plt object!!!
    - [x] extend webapp_gradcam function to work with lime as well
    - check every function to confirm it work with the new setup

- [ ] webapp frontend
    - [x] dynamic model selection
    - [x] fix position of images/buttons (make them "stay put") (optional)
    - [x] add code for upload to webapp server
    - [x] fix image sizes
    - [x] add radio buttons for lime/gradcam
    - [x] add input for server address
    - [x] hide incoming image from server w/ loading animation until server response
    - [x] error handling
        - [x] handle server status
    - [x] server url stuff
    - [ ] Add about page? (EXTREMELY OPTIONAL)

- [ ] webapp server
    - [x] create tester app
    - [x] create gradcam func for single image
    - [x] select model stuff
    - [x] select backend stuff
    - [x] select LIME backend on server (aka need a function like webapp_gradcam but for LIME)

- [ ] Relazione
    - [ ] understand what each lime color mean
    - [ ] read about current SoTA approaches (maybe implement some of the ideas they do?)
    - [ ] Introduction
        - [ ] Write a long ass introduction, it should include (the following tasks)
        - [ ] description of the problem, why is it useful (maybe statistics of internet scammed people estimate etc)
        - [ ] description of some SoTA approaches and how they work

- [ ] Misc
    - [ ] do start script for webapp
    - [ ] Add text for various explanations
    - [ ] solve all the other code TODOs (ctrl+shift+f TODO)

- [ ] REWORK EVERYTHING SO THAT WE HAVE 2 OUTPUT NEURONS INSTEAD OF 1!!!!!!!!!


## Data
https://www.kaggle.com/datasets/cashbowman/ai-generated-images-vs-real-images?resource=download

## Useful links
https://coderzcolumn.com/tutorials/artificial-intelligence/lime-explain-keras-image-classification-network-predictions

https://www.kaggle.com/code/hamedaraab/classification-of-real-and-ai-generated-images#Utilities