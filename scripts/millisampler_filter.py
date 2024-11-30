import pandas as pd
from tqdm import tqdm 


agg_window = 10
agg_stride = 5

if __name__ == '__main__':
    millidf = pd.read_csv(f"data/meta_w{agg_window}_s{agg_stride}_all.csv")
    fltdf = pd.DataFrame(columns=millidf.columns)
    fltdf.columns = millidf.columns
    for i, row in tqdm(millidf.iterrows(), total=millidf.size):
        if max(row[-agg_window:]) >= max(row['inCongestionBytes_aggregate'], row['inRxmitBytes_aggregate']):
            fltdf.loc[i] = row
    fltdf.to_csv(f"data/meta_w{agg_window}_s{agg_stride}_all_filtered.csv", index=False)