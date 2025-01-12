import pandas as pd
import sys


def calculate_overlap_ratio(df1, df2):
    """
    Calculate the overlap ratio between two pandas DataFrames.

    Parameters:
        df1 (pd.DataFrame): The first DataFrame.
        df2 (pd.DataFrame): The second DataFrame.

    Returns:
        float: The overlap ratio.
    """
    #* Ensure the column order is consistent
    df1 = df1.sort_index(axis=1)
    df2 = df2.sort_index(axis=1)
    
    #* Identify the intersection of rows
    common_rows = pd.merge(df1, df2, how='inner')
    num_common = len(common_rows)

    #* Calculate overlap ratio
    total_rows = len(df1) + len(df2)
    overlap_ratio = num_common / total_rows if total_rows > 0 else 0

    return overlap_ratio

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
    
    df1 = df1.drop(columns=['Date first seen', 'Dst IP Addr', 'Src IP Addr', 'Duration', 'Bytes', 'Packets'])
    df2 = df2.drop(columns=['Date first seen', 'Dst IP Addr', 'Src IP Addr', 'Duration', 'Bytes', 'Packets'])
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