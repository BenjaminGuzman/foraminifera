# Foraminifera

Foraminifera species identification using supervised machine learning

### Directory structure

- `img`: directory where all images should go. **DO NOT** upload this directory to GitHub please, use another service like OneDrive or Google Drive. In my case you can download the directory from _url to directory_. The contents of the directory are:
  
  - `species`: Original images dataset, sorted by foraminifera specie. Obtained from https://research.ece.ncsu.edu/aros/foram-identification/
  
  - `segmented`: Segmented images. Images with everything but the foraminifera.
  
  - `segmented_cropped`: Same as `segmented` directory but these images are cropped. **This is the training dataset**.

#### Files

- `segmentation.py`: Code used to generate the `species`, `segmented` and `segmented_cropped` directories.


