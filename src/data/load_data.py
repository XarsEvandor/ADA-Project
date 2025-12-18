import os
import pandas as pd
from ..utils.data_utils import *

def load_data():
    """
    Loads the cleaned DataFrames from pickle files.
    If the pickles don't exist yet, runs preprocessing.main() to create them.
    Ensures all numeric columns are converted to float for consistency.
    """
    # base_dir = os.path.dirname(os.path.abspath(__file__))  
    base_dir = Path(__file__).resolve().parent   
    data_dir = os.path.join(base_dir, "../../data")       
    title_path = os.path.join(data_dir, "full_title_df.pkl")
    body_path = os.path.join(data_dir, "full_body_df.pkl")

    # Launches preprocessing if the pickle file does not exist
    if not (os.path.exists(title_path) and os.path.exists(body_path)):
        print("Pickle files not found — running preprocessing...")
        preprocessing()
        print("Preprocessing done — pickles created.")

    # Loading pickle file
    full_title_df = pd.read_pickle(title_path)
    full_body_df = pd.read_pickle(body_path)

    print("Data successfully loaded from pickle files.")
    return full_title_df, full_body_df

def load_embeddings():
    """
    Loads the parsed Reddit embeddings (users & subreddits) from pickle files.
    If the pickles don't exist yet, runs embeddings_preprocessing() to create them.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))  
    data_dir = os.path.join(base_dir, "../../data")        
    user_path = os.path.join(data_dir, "user_embeddings_df.pkl")
    sub_path  = os.path.join(data_dir, "subreddit_embeddings_df.pkl")

    # Launches preprocessing if the pickle file does not exist
    if not (os.path.exists(user_path) and os.path.exists(sub_path)):
        print("Pickle files not found — running embeddings preprocessing...")
        embeddings_preprocessing()
        print("Preprocessing done — pickles created.")

    # Loading pickle files
    user_df = pd.read_pickle(user_path)
    sub_df  = pd.read_pickle(sub_path)
    
    print("Embeddings successfully loaded from pickle files.")
    return user_df, sub_df