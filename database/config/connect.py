from dotenv import load_dotenv, dotenv_values
from sqlalchemy import create_engine

# load environmental variables
load_dotenv()

# establish connection to PostGIS database
def connect(): 
    config = dotenv_values(".env")
    engine = create_engine(config['CONNECTION_STRING'], echo=False)
    return engine
