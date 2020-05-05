# Disaster Response Pipeline Project

##Project Details
This project predicts the viability of messages received at the time of disasters and classifies the message into 36 categories that help the response teams to decide the appropriate action to take

## Note : 
The model uses GridSearchCV to find the best parameter. But it has been commented out since it takes a lot of time to build the model. User can choose to uncomment the respective portion in models/train_classifier.py file, under build_model function.

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
