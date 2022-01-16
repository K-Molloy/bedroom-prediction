# Street Group Technical Test

#### Kieran Molloy
#### 2021-01-16


## Structure
There are 3 components to this submission
- Exploratory - covers my understanding of the dataset with some added commentary
- Train - contents to create a docker image that can train the model
- Test - contents to create (and deploy) the machine learning model

## Installation

Cloning the repository and getting the datafile
```
git clone git@github.com:K-Molloy/street-group-test.git && cd street-group-test
mkdir data && wget <insert-google-storage-location-here> -O data/raw.json
```
The json file is technically malformed, but each line is a well-formed json object. 

### Exploratory 
```
pipenv install && pipenv shell # make a virtual environment however desired, I quite like pipenv
```

### Training

```
cd train
docker build -t sg-train:1.0.0
```

### Testing / Predicting

```
cd predict
docker built -t sg-predict:1.0.0
```

## Running

### Exploratory
Open an interactive shell, and view - I like Jupyter lab, but any viewer works fine
```
jupyter lab ## or other desired method for viewing ipynb files
```

### Training
Prior to training the dataset must be split in a similar way to that in the exploratory analysis, i.e. 60/30/10 except here it is 80/20 test vs train. So prior to running the docker container
```
cd data
python3 -m split.py
```
This should place 4 files in the `/data` dir, `X_test X_train y_test y_train`.
```
docker run -i -v /<insert-local-path-here>/street-group-test/data:/data -t sg-train:1.0.0
```
Passing this volume to the container allows it to read and write to that location, reading in `X_train` and `X_test`, and outputting 3 files
- `model-params.joblib` - the final variables included
- `preprocessor.joblib` - the parameters used for preprocessing, so they can be replicated later
- `optimal_model.joblib` - the optimal parameters used for modelling, so that predictions can be made


### Testing / Predicting
Ongoing prediction is handled by a tiny fastapi app. Again passing in the model file
```
docker run -i -v /<insert-local-path-here>/street-group/data:/data -p 0.0.0.0:8080:8080 sg-predict:1.0.0
```
This creates a local uvicorn ASGI app, and allows requests to the following endpoints 
- `POST house/predict-bedrooms`
- `POST houses/predict-bedrooms`
- `POST houses/performance-report`

Here are some examples 
```
 curl -X POST http://0.0.0.0:8080/v1/house/predict-bedrooms -H "Content-Type: application/json" -d '{"property_type":"Terraced","total_floor_area":77.0,"number_habitable_rooms":5,"number_heated_rooms":5,"estimated_min_price":184000,"estimated_max_price":224000,"latitude":51.339409,"longitude":0.752466, "number_bedrooms": 2}'
```
But there is also a framework for predicting several sites at once, and requires properly formed JSON.
```
 curl -X POST http://0.0.0.0:8080/v1/houses/predict-bedrooms -H "Content-Type: application/json" -d '[{"property_type":"Terraced","total_floor_area":77.0,"number_habitable_rooms":5,"number_heated_rooms":5,"estimated_min_price":184000,"estimated_max_price":224000,"latitude":51.339409,"longitude":0.752466}]'
```
And the performance report, which returns the accuracy score, confusion matrix and classification report
```
 curl -X POST http://0.0.0.0:8080/v1/houses/performance-report -H "Content-Type: application/json" -d '[{"property_type":"Terraced","total_floor_area":77.0,"number_habitable_rooms":5,"number_heated_rooms":5,"estimated_min_price":184000,"estimated_max_price":224000,"latitude":51.339409,"longitude":0.752466, "number_bedrooms": 2}]'
```
