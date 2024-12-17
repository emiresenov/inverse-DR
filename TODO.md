# TODO
- Figure out how to scale time (should be easy) and PDE terms (seems tricky, need to preserve parameter values or rescale them post-training).
- Figure out why scatterplot on loglog scale cases issues with plotly.



# Completed
- ~~Confirm that Case 1 breaks on long time scale.~~
- ~~For less error-proneness, read voltage constant from utils in Case 0 and Case 1 instead of redefining them in config.py and models.py.~~
- ~~Case 0 eval plot comes out incorrect probably due to loading the wrong model. Fix it.~~ CAUSE: didn't save model in config settings
