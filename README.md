# Disaster Response Pipeline Project

### Project Description:
This UDACITY project 2 has been focused on Natural Language Processing.
Figure Eight company provided data set of messages (tweets and messages) that have been pre-categorized (labeled).
The goal of this project was to build NLP that will categorize messages and make sure that only relevant ones are properly displayed.

### Overview of project steps:
The project has been devided into 3 steps:
1. <b>Data Processing</b> - at this point pipeline (ETL Pipeline) has been created to extract data from the source (disaster_messages.csv), then clean them and save inside proper database and table.
2. <b>Machine Learning</b> - in this stage cleaned and processed data from previous point was used to train model and evaluate it in order to make sure it is sufficiently accurate and can properly categorize the messages
3. <b>The Web App</b> - this is to show how the model works in real time

### Installation:
''' git clone https://github.com/Soundoffear/DS_Nano_Disaster_response.git '''

### Instructions on how to execute program:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/


### Files in this notebook:
* app/templates/go.html - file for web app
* app/templates/master.html - file for web app
* app/run.py - python file responsible for running web app

* data/DisasterResponse.db - database of cleaned messages
* data/disaster_categories.csv - provided file of classified messages
* data/disaster_messages.csv - provided file of messages
* data/process_data.py - file with all the code to process data

* models/train_classifier.py - file with all the code to train your classifier

* README.md - this description

### Libraries used:
- pandas - library for data processing
- numpy - library for numerical data processing
- sklearn - library for machine learning
- sqlalchemy - library for reading and writing into SQL databases