import os
import pandas as pd
from tqdm import tqdm
from glob import glob
from rich import print as pprint
from multiprocessing import Pool, cpu_count

# Column definitions: https://github.com/microsoft/Yatesbury?tab=readme-ov-file#dataset-description
nsg_columns = [
    "Timestamp", "SrcIp", "DstIp", "SrcPt", "DstPt", "Proto", "FlowDir",
    "Decision", "FlowState", "PktSent", "BytesSent", "PktRecv", "BytesRecv"
]
label_columns = ["SrcIp", "DstIp", "StartTime", "Label"]

# Paths
output_dir = os.path.expanduser('~/Desktop/Datasets/Yatesbury/labeled')
os.makedirs(output_dir, exist_ok=True)

files = glob('/Users/hongyu/Desktop/Datasets/Yatesbury/data/*/*')
triplet_groups = [files[i:i+3] for i in range(0, len(files), 3)]
# pprint(triplet_groups)


def process_triplet(triplet):
    graph_dir, label_path, nsg_path = triplet
    attack_name: str = os.path.basename(os.path.dirname(label_path))
    if 'normal_' in attack_name:
        attack_name = attack_name.replace('normal_', '')
    if 'attack_' in attack_name:
        attack_name = attack_name.replace('attack_', '')
    if '_attacks' in attack_name:
        attack_name = attack_name.replace('_attacks', '')

    try:
        # Read NSG data
        nsgdf = pd.read_csv(nsg_path)
        nsgdf.columns = nsg_columns
        nsgdf['Timestamp'] = pd.to_datetime(nsgdf['Timestamp'], errors='coerce')
        nsgdf.fillna(0, inplace=True)
        nsgdf['Label'] = 0

        # Read label data
        labeldf = pd.read_csv(label_path)
        labeldf.columns = label_columns
        labeldf['StartTime'] = pd.to_datetime(labeldf['StartTime'], errors='coerce')
        labeldf['EndTime'] = labeldf['StartTime'] + pd.Timedelta(minutes=2)

        # Labeling
        for _, row in tqdm(labeldf.iterrows(), total=labeldf.shape[0], desc=f"Labeling {attack_name}"):
            mask = (
                (nsgdf['SrcIp'] == row['SrcIp']) &
                (nsgdf['DstIp'] == row['DstIp']) &
                (nsgdf['Timestamp'] >= row['StartTime']) &
                (nsgdf['Timestamp'] < row['EndTime'])
            )
            nsgdf.loc[mask, 'Label'] = row['Label']

        # Save individual file
        labeled_path = os.path.join(output_dir, f'{attack_name}.csv')
        nsgdf.to_csv(labeled_path, index=False)

        # Add Attack column and return
        nsgdf['Attack'] = attack_name
        return nsgdf

    except Exception as e:
        print(f"[Error] Failed processing {attack_name}: {e}")
        return None


if __name__ == '__main__':
    with Pool(processes=5) as pool:
        results = list(tqdm(pool.imap(process_triplet, triplet_groups), total=len(triplet_groups)))

    # Filter out any None results (failures)
    all_dataframes = [df for df in results if df is not None]

    # Combine and save full CSV
    combined_df = pd.concat(all_dataframes, ignore_index=True)
    combined_df.to_csv(os.path.join(output_dir, 'yatesbury_all.csv'), index=False)