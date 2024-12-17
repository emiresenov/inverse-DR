# TODO
- Figure out how to scale time (should be easy) and PDE terms (seems tricky, need to preserve parameter values or rescale them post-training?). Actually, do I need to rescale? Can't I just compress my solution to [0,1] scale?
- Figure out why scatterplot on loglog scale cases issues with plotly.

# Completed
- ~~Confirm that Case 1 breaks on long time scale.~~
- ~~For less error-proneness, read voltage constant from utils in Case 0 and Case 1 instead of redefining them in config.py and models.py.~~
- ~~Case 0 eval plot comes out incorrect probably due to loading the wrong model. Fix it.~~ **CAUSE**: didn't save model in config settings.





## Idea for rescaling
- Rescale time values by dividing each time value by t_max.
- Rescale solution by dividing each current value by the max current.
- In models.py, store max current as a val, use this to rescale PDE loss.

### TODO, try rescaling solution on Case 1
**How do we rescale the PDE?**
 - Idea rescale all non-scaled terms (everything not including \hat{u}) by u_max.
 - TRY AFTER LUNCH

 **Issue: scaling is not proportional**
- TODO: Verify by not scaling the variables and see if we get a perfect prediction function. 
- Correction: even if we don't scale variables, proportionality issue still exists since the variables are scaled coming in. This means that the initial condition has the wrong scale, as well as the residual.




# Crazy idea
What if we just transform everything to a log-log scale?
**Problem:** inverse parameters are still on a massive scale, making us dependent on good starting guesses.

I think figuring out good normalization method is the way to go.