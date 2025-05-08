import pandas as pd
from pathlib import Path

# Path to the data folder
DATA_DIR = Path(__file__).resolve().parents[1] / "data"

def load_raw(filename: str, sep: str = ",") -> pd.DataFrame:
    """
    Load a raw data file from data/ and return a DataFrame.
    
    Parameters:
        filename (str): Name of the CSV file in the data/ folder.
        sep (str): Delimiter used in the CSV file. Default is ','.
    
    Returns:
        pd.DataFrame: Loaded DataFrame.
    """
    path = DATA_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"{path} not found.")
    return pd.read_csv(path, sep=sep)

if __name__ == "__main__":
    # Example usage for semicolon-separated file
    df = load_raw("data-corrupted.csv", sep=";")
    print(f"Shape: {df.shape}")
    print(df.head())
    
    
