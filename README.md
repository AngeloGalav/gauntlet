# gauntlet

<p align="center">
  <img src="res//image.png" alt="Data Thanos" width="400px"/>
</p>

> _"I used the AI to destroy the AI"_
\- Thanos if he was a data scientist

GAUNTLET is an explainable model for detecting AI-generated visual
content. By examining the finer characteristics of AI generated images (such as noise patterns), it can distinguish AI-generated images from human-crafted ones. In addition, it outputs a heatmap which highlights the features that activated the model.

Our aim is not just to provide a tool for people to use, but also to educate them into learning the characteristics of an AI generated image, so that they can detect and avoid malicious content without the necessity of this app. In a sense, the app should train the user so that it won't rely on this app. Kinda crazy if you think about it.

Click the following image to watch a video demo of GAUNTLET:
[![Watch the video](https://img.youtube.com/vi/THj-Gn8MYkw/maxresdefault.jpg)](https://www.youtube.com/watch?v=THj-Gn8MYkw)

More details can be found in `report.pdf` or in the notebooks.

## How to use:
To use the Notebooks and the WebApp, you'll need to first create python environment and install the required packages using pip:
1. `python -m venv .venv`
2. On Linux: `source .venv/bin/activate`
2b. On Windows: `.venv/Scripts/activate`
3. `pip install -r requirements.txt`

### Notebooks
Before running the notebooks, place the respective datasets folders in the data folder:
- CIFAKE contents into CIFAKE folder
- RealVsAiArt contents into RealVsAiArt folder

### WebApp
To run the webapp, use the `./start_webapp` script. If for some reason it doesn't work, you can instead use the following commands:
```
python webapp/server.py
cd webapp/frontend
npm install
npm start
```

## Datasets used
- [RVAA](https://www.kaggle.com/datasets/cashbowman/ai-generated-images-vs-real-images?resource=download)
- [CIFAKE](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images)