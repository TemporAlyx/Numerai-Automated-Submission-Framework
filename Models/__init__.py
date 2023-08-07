import os

models = [x[:-3] for x in os.listdir(os.path.dirname(__file__)) 
          if x.endswith('.py') and x != '__init__.py' and 'utils' not in x]
for model in models:
    exec('from Models import {}'.format(model))