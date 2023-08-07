# Numerai-Automated-Submission-Framework
 A simple (unofficial) framework for automating model submissions for the Numerai tournament.

Once setup by adding your numerai account id and key to the config.json file, simply running auto-framework.py will check for and download round data, and then process all the models found within the /Models folder. 

An example LightGBM model is provided, both the notebook used to train and save the model, as well as the Models/LightGBM.py file, demonstrating how to configure the CustomModel class for automatic predictions.

While not overly complex, and perhaps a little rough around the edges, this is the system I have been using to train and submit my models, with a task scheduled daily to run auto-framework.py. The main point was to make it easy to drop in new models, as well as painlessly add variations of a model. (testing degrees of feature neutralization, or various model combinations)

My data pipeline and diagnostics code may seem a little strange if you are used to the numerai example notebooks, as I rewrote most things to rely more on numpy than pandas. Mostly this was just due to what I was familiar when I started out, and then stuck around due to ~~technical debt~~ personal preference.

I don't believe this would be too difficult to automate in the cloud, although I know others already have tutorials for such solutions, my hope is that this can simply be another repository of example code for newcomers to the tournament.

### Future Plans
- look into more advanced task scheduling to avoid missing late round opens
- add support for uploading numerbay submissions
- add more example models, and refactor/comment more code for increased readability
- benchmark diagnostic functions and maximize performance
