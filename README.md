# ethics_project2
Pending

How to use:
Place the respective datasets folders in the data folder
(i.e. CIFAKE in CIFAKE folder)
cant commit those since they are too big...

## TODOs
Plz keep the todo list updated!
- [x] Create some models
    - [ ] test standard CNN
    - [x] Fine-tuned resnet
    - [ ] Implement early stopping?
- [x] Rewrite to FTModel class to avoid call of model.model (using only single model instead)
- [x] test models
- [ ] do Xai stuff
    - [x] find a method for XAI (LIME?)
    - [x] fix LIME (batch_predict)
- [ ] Add more graphs?
- [x] better Finetuning of resnet50 (fine tune feature extractor after finetuning classifier)

- [ ] test with new image dataset
    - [ ] Add transformation pipeline
    - [ ] 
- [ ] Misc
    - [ ] save graph and data (e.g. csv) to files for easier comparison
    - [ ] solve all the other code TODOs (ctrl+shift+f TODO)

- [ ] Wait for professor's new directives

## Useful links
https://coderzcolumn.com/tutorials/artificial-intelligence/lime-explain-keras-image-classification-network-predictions

https://www.kaggle.com/code/hamedaraab/classification-of-real-and-ai-generated-images#Utilities