# Foraminifera

Foraminifera species identification using supervised machine learning


## Directory structure

- `img`: directory where all images should go. **DO NOT** upload this directory to GitHub please, use another service like OneDrive or Google Drive. In my case you can download the directory from [https://correoipn-my.sharepoint.com/:f:/g/personal/bvelascog1600_alumno_ipn_mx/EtCCVtNMvKpGuzhYBfVhdUQBJ46_8a7QQg0xZI5eZQz0QA?e=b3sWva](https://correoipn-my.sharepoint.com/:f:/g/personal/bvelascog1600_alumno_ipn_mx/EtCCVtNMvKpGuzhYBfVhdUQBJ46_8a7QQg0xZI5eZQz0QA?e=b3sWva). The descriptions of the contents are:
  
  - `species`: Original images dataset, sorted by foraminifera specie. Obtained from [https://research.ece.ncsu.edu/aros/foram-identification/](https://research.ece.ncsu.edu/aros/foram-identification/)
  
  - `segmented`: Segmented images. Images only with the foraminifera.
  
  - `segmented_cropped`: Same as `segmented` directory but these images are cropped. **This is the training dataset**.

## Files

- `segmentation.py` file obtains the training dataset, in most cases it works really well, but sometimes it just do segmentation wrong. See GitHub issues for details.
- `data_exploration.ipynb` Is a notebook that contains the exploration of the training dataset. From this exploration I conclude features for the identification are similar between species, hence it's necessary to change the feature extraction algorithm or to train with the dataset as it is, without changing anything, and using deep learning.
- `pca.ipynb` Another notebook that actually contains the NN architecture (or at least what I've tried so far). Maybe the NN can give better results if the preprocessing is better. Also, maybe having the images rotated can affect the algorithm, that's why I created
- `rectifier_algorithm.ipynb` A rectifier algorithm to rotate all images and improve the NN performance. Or at least that's the idea, but in practice the NN does not have a better performance.

## Data flow

As probably guessed, the data flow should look like this

**segmentation.py -> data\_exploration.ipynb -> pca.ipynb**