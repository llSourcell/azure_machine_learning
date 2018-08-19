# This script generates the scoring and schema files
# necessary to operationalize your model
from azureml.api.schema.dataTypes import DataTypes
from azureml.api.schema.sampleDefinition import SampleDefinition
from azureml.api.realtime.services import generate_schema
from sklearn.externals import joblib
import pandas as pd

# Prepare the web service definition by authoring
# init() and run() functions. Test the functions
# before deploying the web service.

model = None

def init():
    # load the model file
    global model
    model = joblib.load('model.pkl')

def run(input):
    prediction = model.predict(input)
    print(prediction)
    return str(prediction)

def generate_api_schema():
    import os
    print("create schema")
    input=6.0
    inputs = {"input": SampleDefinition(DataTypes.STANDARD, input)}
    os.makedirs('outputs', exist_ok=True)
    print(generate_schema(inputs=inputs, filepath="outputs/schema.json", run_func=run))

# Implement test code to run in IDE or Azure ML Workbench
if __name__ == '__main__':
    # Import the logger only for Workbench runs
    from azureml.logging import get_azureml_logger

    logger = get_azureml_logger()

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--generate', action='store_true', help='Generate Schema')
    args = parser.parse_args()

    print(args)
    #forecfully setting the generate to true
    args.generate=True
    if args.generate:
        print('generating api schema')
        generate_api_schema()

    init()
    input = 6.1
    result = run(input)
    logger.log("Result",result)
