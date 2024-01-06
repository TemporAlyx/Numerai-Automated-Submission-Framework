# Numerai-Automated-Submission-Framework
 (wouldn't mind a better / shorter name for this)
 A simple (unofficial) framework for automating model submissions for the Numerai tournament.

Once setup by adding your numerai account id and key to the config.json file (an empty config will be made on first run), simply running auto-framework.py will check for and download round data, and then process all the models found within the /Models folder. 

An example LightGBM model is provided, both the notebook used to train and save the model, as well as the Models/LightGBM.py file, demonstrating how to configure the CustomModel class for automatic predictions.

While not overly complex, and perhaps a little rough around the edges, this is the system I have been using to train and submit my models, with a task scheduled daily to run auto-framework.py. The main point was to make it easy to drop in new models, as well as painlessly add variations of a model. (testing degrees of feature neutralization, or various model combinations)

My data pipeline and diagnostics code may seem a little strange if you are used to the numerai example notebooks, as I rewrote most things to rely more on numpy than pandas. Mostly this was just due to what I was familiar when I started out, and then stuck around due to ~~technical debt~~ personal preference.

I don't believe this would be too difficult to automate in the cloud, although I know others already have tutorials for such solutions, my hope is that this can simply be another repository of example code for newcomers to the tournament.

### Future Plans
- add support for uploading numerbay submissions
- add logging and email or sms notifications for pipeline failures / successes
- extend functions for validation models to make fancy graphs
- add more example models, and refactor/comment more code for increased readability
- look into switching diagnostics functions over to numerai-tools

### My Models
If you would like to see my Numerai models here is a list of links to each of their pages: [MesozoicMetallurgist Numerai Model List](https://docs.google.com/document/d/19P_e8ahJUr6HbaOfFAmJSLXcXZWgzSj3lf_zyecBatM/edit?usp=sharing)
