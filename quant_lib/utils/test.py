import pandas as pd

def test_file_columns(file_path):
    data  = pd.read_parquet(file_path)
    print(f"data columns--->{data.columns}")

if __name__ == '__main__':
    test_file_columns()