import pandas as pd
import sys
import hashlib


def hash_row(row):
    """
    Hash a row to a unique string using SHA256.
    """
    return hashlib.sha256(pd.util.hash_pandas_object(row).values.tobytes()).hexdigest()

def calculate_overlap_ratio(df1, df2):
    """
    Calculate the overlap ratio between two pandas DataFrames, optimized for large datasets.

    Parameters:
        df1 (pd.DataFrame): The first DataFrame.
        df2 (pd.DataFrame): The second DataFrame.

    Returns:
        float: The overlap ratio.
    """
    # Hash rows to unique identifiers for each DataFrame
    df1_hashed = df1.apply(hash_row, axis=1)
    df2_hashed = df2.apply(hash_row, axis=1)

    # Convert hashed rows to sets
    set1 = set(df1_hashed)
    set2 = set(df2_hashed)

    # Calculate intersection size and overlap ratio
    common_count = len(set1 & set2)
    total_count = len(set1) + len(set2)
    
    return common_count / total_count if total_count > 0 else 0

if __name__ == "__main__":
    # #* Sample DataFrames
    # data1 = {'A': [1, 2, 3], 'B': [4, 5, 6]}
    # data2 = {'A': [2, 3, 4, 1], 'B': [5, 6, 7, 8]}

    # df1 = pd.DataFrame(data1)
    # df2 = pd.DataFrame(data2)
    
    path1 = sys.argv[1]
    path2 = sys.argv[2]
    df1 = pd.read_csv(path1)
    df2 = pd.read_csv(path2)
    
    df1 = df1.drop(columns=['Date first seen', 'Dst IP Addr', 'Src IP Addr', 'Bytes', 'Packets'])#, 'Duration'])
    df2 = df2.drop(columns=['Date first seen', 'Dst IP Addr', 'Src IP Addr', 'Bytes', 'Packets'])#, 'Duration', 'Bytes', 'Packets'])
    if 'Flows' in df1.columns:
        df1 = df1.drop(columns=['Flows'])
    if 'Flows' in df2.columns:
        df2 = df2.drop(columns=['Flows'])
    if 'Src Pt' in df1.columns:
        df1 = df1.drop(columns=['Src Pt'])
    if 'Src Pt' in df2.columns:
        df2 = df2.drop(columns=['Src Pt'])
    
    print(f"Loaded {len(df1)} rows from {path1}")
    print(f"Loaded {len(df2)} rows from {path2}")

    ratio = calculate_overlap_ratio(df1, df2)
    print(f"Overlap Ratio: {ratio: .3f}")