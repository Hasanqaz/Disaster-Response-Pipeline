# Project Description
Figure Eight has given a data set containing real messages that were sent during disaster events. In this project, I have created a machine learning pipeline to categorize these events, so that the messages can be sent to an appropriate disaster relief agency.

# Installation
Anaconda distribution of Python and NLTK library (including the packages: punkt, wordnet, averaged_perceptron_tagger)

# File Descriptions
- App folder including the templates folder and "run.py" for the web application
- Data folder containing "DisasterResponse.db", "disaster_categories.csv", "disaster_messages.csv" and "process_data.py" for - data cleaning and transfering.
- Models folder including "classifier.pkl" and "train_classifier.py" for the Machine Learning model.
- README file

# Instructions
1. Run the following commands in the project's root directory to set up your database and model:
    - To run ETL pipeline that cleans data and stores in database python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
    - To run ML pipeline that trains classifier and saves python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
2. Run the following command in the app's directory to run your web app. python run.py
3. Go to http://0.0.0.0:3001/

# Licensing, Authors, Acknowledgements
Must give credit to Figure Eight for the data and Udacity for giving the needed training to make this happen. Feel free to use the code here as you would like!