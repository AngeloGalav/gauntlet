# gauntlet
> _"I used the AI to destroy the AI"_
\- Thanos if he was a data scientist

Pending

## How to use:
Place the respective datasets folders in the data folder:
- CIFAKE contents into CIFAKE folder
- RealVsAiArt contents into RealVsAiArt folder

## TODOs
Plz keep the todo list updated!
- [ ] IMPORTANT!!!! NEED TO FIX LABELS FOR IMAGES (ESTABLISH WHAT 0 AND 1 MEAN!!!!!!!!!!)
- [x] Create some models
    - [ ] test standard CNN
    - [x] Fine-tuned resnet
    - [ ] Implement early stopping?
- [x] Rewrite to FTModel class to avoid call of model.model (using only single model instead)
- [x] test models
- [ ] do Xai stuff
    - [x] find a method for XAI (LIME?)
    - [x] fix LIME (batch_predict)
    - [ ] try CAPTUM? (https://captum.ai/)
- [ ] Add more graphs?
- [x] better Finetuning of resnet50 (fine tune feature extractor after finetuning classifier)
- [x] test with new image dataset
    - [x] Add transformation pipeline
    - [x] Add code for new image
- [ ] Misc
    - [ ] save graph and data (e.g. csv) to files for easier comparison
    - [ ] Add text for various explanations
    - [ ] solve all the other code TODOs (ctrl+shift+f TODO)

- [ ] Wait for professor's new directives

## Data
https://www.kaggle.com/datasets/cashbowman/ai-generated-images-vs-real-images?resource=download

## Useful links
https://coderzcolumn.com/tutorials/artificial-intelligence/lime-explain-keras-image-classification-network-predictions

https://www.kaggle.com/code/hamedaraab/classification-of-real-and-ai-generated-images#Utilities