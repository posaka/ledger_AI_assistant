from dotenv import load_dotenv
load_dotenv() 
import os

print(os.getenv("EMBEDDINGS_PROVIDER"))