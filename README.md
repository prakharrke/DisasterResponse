# Disaster Response Pipeline Project

##Project Details

### Summary: 
This project predicts the viability of messages received at the time of disasters and classifies the message into 36 categories that help the response teams to decide the appropriate action to take.

The motivation of the project comes from Figure Eight, who are trying to solve a very real problem of understanding the viability of messages during disasters. Having to comb through every single message by the disaster response teams is a real challange, not to mention the time it consumes to manually do so. Hence, in order to make message classification received during such times simpler, we have built this model.

The model uses RandomForestClassifier. A random forest is a meta estimator that fits a number of decision tree classifiers on various sub-samples of the dataset and uses averaging to improve the predictive accuracy and control over-fitting. 

The model has been trained on the data provided by Figure Eight. They have labelled this data, collected from real life disaster scenarios.

The accuracy achieved for this model is 94.90%

### Project Structure

data folder contains all the python and csv files for data and process_data.py file which performs ETL operations. 

models folder contains train_classifer.py file which builds, trains and evaluates the model and creates a classifier.pkl file.

app folder houses the code for web app. run.py file is responsible for starting up the web app. It uses Flask. Template folder contains all the required HTML files.


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
