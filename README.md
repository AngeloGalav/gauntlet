# gauntlet
> _"I used the AI to destroy the AI"_
\- Thanos if he was a data scientist

Pending

## How to use:
Place the respective datasets folders in the data folder:
- CIFAKE contents into CIFAKE folder
- RealVsAiArt contents into RealVsAiArt folder

## TODOs
- [x] Create some models
    - [ ] test standard CNN
    - [x] Fine-tuned resnet
    - [x] Implement confusion matrix code
    - [x] Use mixed data model
    - [ ] Implement data augmentation in Mixed model (maybe turn data augmentation dataset creation step into a function?)
    - [ ] Fix Mixed Dataset model (accuracy stuck at 34.5%)
- [x] Rewrite to FTModel class to avoid call of model.model (using only single model instead)
- [x] test models
- [ ] do Xai stuff
    - [x] find a method for XAI (LIME?)
    - [x] fix LIME (batch_predict)
    - [ ] try CAPTUM? (https://captum.ai/)
    - [ ] Combining Grad-CAM with Guided Backpropagation (?)
        - tried it, failed miserably (needs grad to be computed or smth)
    - [x] save gradcam/lime result (i.e. outputs/model_name/grad_cam/CIFAKE or outputs/model_name/lime/RVAA etc...)
    - [x] output prediction probability (useful if we make webapp)
    - [ ] batch lime/gradcam (no display, only save) (i.e. run and save all gradcam for a batch or collection of batches to see them later)
    - [x] test to see if we can use less layers in gradcam for speeding up the computation (result: layer4 should be fine for resnet).
    - [ ] create gradcam/lime function that takes a SINGLE image as input (in the format found in server.py) and returns the plt object!!!
- [x] Plotting
    - [x] f1, precision, recall graphs
    - [x] update LIME function to show predicted/label etc...
- [x] better Finetuning of resnet50 (fine tune feature extractor after finetuning classifier)
- [x] test with new image dataset
    - [x] Add transformation pipeline
    - [x] Add code for new image
    - [x] fix data augmentation
- [x] Misc
    - [x] save graph and data (e.g. csv) to files for easier comparison
        - [x] Fix report.txt bug (sometimes it does not get saved??? also there's on in outputs/report...)
    - [x] add display mode (meaning, add a boolean/var so that it can be set to off or on when displaying plots on the notebook, so that the notebook is not cluttered, at least not in this phase of dev)
    - [ ] Add text for various explanations
    - [ ] solve all the other code TODOs (ctrl+shift+f TODO)

- [ ] webapp fronted
    - [ ] create frontend
    - [ ] create dropdown menu with models
    - [ ] image input
    - [ ] display image output
- [ ] webapp server
    - [x] create tester app
    - [ ] create gradcam func for single image

## Data
https://www.kaggle.com/datasets/cashbowman/ai-generated-images-vs-real-images?resource=download

## Useful links
https://coderzcolumn.com/tutorials/artificial-intelligence/lime-explain-keras-image-classification-network-predictions

https://www.kaggle.com/code/hamedaraab/classification-of-real-and-ai-generated-images#Utilities