import pandas as pd


if __name__ == '__main__':
  floats = ['Duration']
  ints = ["Packets", "Flows", "Src Pt", "Dst Pt"]
  cid3df = pd.read_csv('data/CIDDS-001/traffic/OpenStack/CIDDS-001-internal-week3.csv').iloc[:, :11]
  cid3df["Proto"] = cid3df["Proto"].str.strip()
  cid3df[ints] = cid3df[ints].astype(int)
  cid3df[floats] = cid3df[floats].astype(float)
  cid3df['Bytes'] = cid3df['Bytes'].apply(lambda cell: int(float(cell.split('M')[0])*1e6) 
                                              if isinstance(cell, str) and 'M' in cell else int(cell))
  cid3df.to_csv('data/cidds_wk3_processed.csv', index=False)