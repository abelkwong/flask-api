import pickle
import pandas as pd
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel

app = FastAPI()

# define input data format using Pydantic
class BanknoteInput(BaseModel):
    variance: float
    skewness: float
    curtosis: float
    entropy: float


# load the model
with open('classifier.pkl', 'rb') as f:
    classifier = pickle.load(f)

# define root endpoint


@app.get('/')
def welcome():
    return "Welcome All"

# define prediction endpoint


@app.post('/predict')
async def predict_note_authentication(data: BanknoteInput):
    """Let's Authenticate the Banks Note
    This is using docstrings for specifications.
    ---
    parameters:
        - name: data
          in: body
          required: true
          schema:
            $ref: '#/definitions/BanknoteInput'
          
    responses:
        200: 
            description: The output values
    """
    # get the data from the input
    input_data = [[data.variance, data.skewness, data.curtosis, data.entropy]]
    # make the prediction
    prediction = classifier.predict(input_data)
    return {"predicted_value": prediction[0]}

# define file upload endpoint


@app.post('/predict_file')
async def predict_note_file(file: UploadFile):
    """Let's Authenticate the Bank Notes
    This is using docstrings for specifications.
    ---
    parameters:
        - name: file
          in: formData
          type: file
          required: true
    
    responses:
        200:
            description: The output values
    """
    # read the data from the file
    df_test = pd.read_csv(file.file)
    # make the prediction
    prediction = classifier.predict(df_test)
    return {"predicted_values": list(prediction)}

# define the data model for the input


@app.get('/docs', include_in_schema=False)
def override_swagger():
    return RedirectResponse(url='/docs')


@app.get('/redoc', include_in_schema=False)
def override_redoc():
    return RedirectResponse(url='/redoc')