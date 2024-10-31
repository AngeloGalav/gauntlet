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
- [x] Rewrite to FTModel class to avoid call of model.model (using only single model instead)
- [x] test models
- [ ] do Xai stuff
    - [x] find a method for XAI (LIME?)
    - [x] fix LIME (batch_predict)
    - [ ] try CAPTUM? (https://captum.ai/)
    - [ ] Combining Grad-CAM with Guided Backpropagation (?)
    - [x] save gradcam/lime result (i.e. outputs/model_name/grad_cam/CIFAKE or outputs/model_name/lime/RVAA etc...)
    - [x] output prediction probability (useful if we make webapp)
- [x] Plotting
    - [x] f1, precision, recall graphs
    - [x] update LIME function to show predicted/label etc...
- [x] better Finetuning of resnet50 (fine tune feature extractor after finetuning classifier)
- [x] test with new image dataset
    - [x] Add transformation pipeline
    - [x] Add code for new image
    - [x] fix data augmentation
    - [ ] RANDOM ASS PERFORMANCE DEGRATION IN RVAA?????????
    - [ ] set SEED better (galf ignora)
- [x] Misc
    - [x] save graph and data (e.g. csv) to files for easier comparison
        - [x] Fix report.txt bug (sometimes it does not get saved??? also there's on in outputs/report...)
    - [x] add display mode (meaning, add a boolean/var so that it can be set to off or on when displaying plots on the notebook, so that the notebook is not cluttered, at least not in this phase of dev)
    - [ ] Add text for various explanations
    - [ ] solve all the other code TODOs (ctrl+shift+f TODO)

## Data
https://www.kaggle.com/datasets/cashbowman/ai-generated-images-vs-real-images?resource=download

## Useful links
https://coderzcolumn.com/tutorials/artificial-intelligence/lime-explain-keras-image-classification-network-predictions

https://www.kaggle.com/code/hamedaraab/classification-of-real-and-ai-generated-images#Utilities