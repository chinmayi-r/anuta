from glob import glob
import gzip
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
import sys


agg_window = 50
agg_stride = agg_window
ctx_len = 6 
ctx_len -= 1 #* The aggregate itself is already counted as context
key_cols = ['rackid', 'hostid']
agg_cols = [
    'IngressBytesAgg', 'EgressBytesAgg', 'InRxmitBytesAgg',
    'OutRxmitBytesAgg', 'InCongestionBytesAgg', 'ConnectionsAgg'
]

def aggregate_timeseries(data, window, stride, func, integer=True):
    """
    Aggregates a time series using a provided function over a sliding window.

    Parameters:
        data (list or np.ndarray): The time series data.
        window (int): The size of the window for aggregation.
        stride (int): The step size for the sliding window.
        func (callable): A function to aggregate the data within each window.
        integer (bool): Whether to convert aggregated results to integers.

    Returns:
        list: The aggregated values for each window.
    """
    if len(data) < window:
        raise ValueError("The length of the data must be greater than or equal to the window size.")
    #* Cut the residual out
    data = data[: len(data) - (len(data)%window)]
    aggregated = []
    for i in range(0, len(data) - window + 1, stride):
        window_data = data[i:i + window]
        result = func(window_data)
        if integer:
            result = int(result)
        aggregated.append(result)
    
    return aggregated

if __name__ == '__main__':
    traces = glob(f"data/Millisampler-data/day1-h1-zip/*")
    print(f"{len(traces)=}")
    
    istest = False if sys.argv[1] == 'train' else True
    # rack_range = range(0, 150) #* Training
    test_range = [156, 158, 160, 172, 177, 178, 179, 181, 184, 191] #* Testing (10 racks)
    # rack_range = range(0, 17000) #* All
    # test_range = [200]
    millidata = []
    ctxdata = []
    rackids = set()
    numracks_limit = 1e100
    hostids = list()
    for trace in tqdm(traces):
        rackid = int(trace.split('rackId_')[-1].split('_')[0])
        if istest and rackid not in test_range: continue
        if not istest and rackid in test_range: continue
        
        rackids.add(rackid)
        if len(rackids) > numracks_limit:
            print(f"Reached the limit of {numracks_limit} racks.")
            break
        
        f = gzip.open(trace)
        record = json.loads(f.read())
        if not 'ingressBytes' in record: continue
        
        hostids.append(record['server_hostname'])
        aggregates = {
            'rackid': rackid,
            'hostid': record['server_hostname'],
            # # 'sampling_freq_ms': record['sampling_freq'], 
            # 'aggregate_window': agg_window,
            # 'aggregate_stride': agg_stride,
            'IngressBytesAgg': aggregate_timeseries(record['ingressBytes'], agg_window, agg_stride, sum, True),
            'EgressBytesAgg': aggregate_timeseries(record['egressBytes'], agg_window, agg_stride, sum, True),
            'InRxmitBytesAgg': aggregate_timeseries(record['inRxmitBytes'], agg_window, agg_stride, sum, True),
            'OutRxmitBytesAgg': aggregate_timeseries(record['outRxmitBytes'], agg_window, agg_stride, sum, True),
            'InCongestionBytesAgg': aggregate_timeseries(record['inCongestionExperiencedBytes'], agg_window, agg_stride, sum, True),
            'ConnectionsAgg': aggregate_timeseries(record['connections'], agg_window, agg_stride, sum, True),
        }
        keys_to_zip = [k for k, v in aggregates.items() if isinstance(v, list)]
        flattened = [
            {**{k: aggregates[k] for k in ['rackid', 'hostid']}, 
             **dict(zip(keys_to_zip, values))}
            for values in zip(*[aggregates[key] for key in keys_to_zip])
        ]
        
        record_len = len(record['ingressBytes'])
        ingressbytes = record['ingressBytes'][: record_len - (record_len % agg_window)]
        # Create context for aggregate keys
        for i, aggregate in enumerate(flattened):
            start = i * agg_stride
            total_bytes = 0
            for j, val in enumerate(ingressbytes[start: start + agg_window]):
                col = f"IngressBytes{j}"
                aggregate[col] = val
                total_bytes += val
            assert total_bytes == aggregate['IngressBytesAgg'], f"Aggregation {i}: {total_bytes=} ≠ {aggregate['IngressBytesAgg']}"

            # Add context features
            ctx = {}
            for k in range(ctx_len, 0, -1):
                if i - k < 0:
                    prev_agg = {col: 0 for col in agg_cols}
                else:
                    prev_agg = flattened[i - k]
                for col in agg_cols:
                    name = col.replace('Agg', f'Ctx') + str(ctx_len - k)
                    ctx[name] = prev_agg[col]
            ctxdata.append(ctx)
            millidata.append(aggregate)
    
    print(f"Processed {len(hostids)} ({len(set(hostids))}) hosts.")
    millidf = pd.DataFrame.from_dict(millidata)
    ctxdf = pd.DataFrame.from_dict(ctxdata)
    keydf = pd.DataFrame(millidf[key_cols])
    millidf = millidf.drop(columns=key_cols)
    print(f"Aggregation shape: {millidf.shape}, context shape: {ctxdf.shape}")
    millidf = pd.concat([keydf, ctxdf, millidf], axis=1)
    
    millidf = millidf.apply(pd.to_numeric, errors='coerce')
    millidf = millidf.fillna(0)  # Fill NaN values with 0
    millidf = millidf.clip(lower=0)  # Ensure no negative values
    millidf = millidf.sort_values(by=['rackid', 'hostid'], kind='stable')
    millidf = millidf.reset_index(drop=True)
    print(f"Aggregated data shape: {millidf.shape}")
    
    label = 'test' if istest else 'train'
    label += f"_{len(rackids)}racks_{ctx_len}ctx"
    savepath = f"data/metadc_{label}.csv"
    millidf.to_csv(savepath, index=False)
    print(f"Saved aggregated data to {savepath}")
    
    rackids = sorted(list(rackids))
    print(f"{len(rackids)} racks")
    # print(f"Rack IDs: \n{rackids}")