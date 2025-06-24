from glob import glob
import gzip
import json
import numpy as np
import pandas as pd
from tqdm import tqdm


agg_window = 50
agg_stride = agg_window

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
    
    istest = False
    # rack_range = range(0, 150) #* Training
    rack_range = [156, 158, 160, 172, 177, 178, 179, 181, 184, 191] #* Testing (10 racks)
    # rack_range = range(0, 17000) #* All
    millidata = []
    rackids = set()
    numracks_limit = 500
    for trace in tqdm(traces):
        rackid = int(trace.split('rackId_')[-1].split('_')[0])
        if istest and rackid not in rack_range: continue
        if not istest and rackid in rack_range: continue
        
        rackids.add(rackid)
        if len(rackids) > numracks_limit:
            print(f"Reached the limit of {numracks_limit} racks.")
            break
        
        f = gzip.open(trace)
        record = json.loads(f.read())
        if not 'ingressBytes' in record: continue
        
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
        for i, aggregate in enumerate(flattened):
            start = i*agg_stride
            total_bytes = 0
            for j, val in enumerate(ingressbytes[start: start+agg_window]):
                key = f"IngressBytes{j}"
                aggregate[key] = val
                total_bytes += val
            assert total_bytes == aggregate['IngressBytesAgg'], f"Aggregation {i}: {total_bytes=} ≠ {aggregate['IngressBytesAgg']}"
            millidata.append(aggregate)
    
    millidf = pd.DataFrame.from_dict(millidata)
    millidf = millidf.apply(pd.to_numeric, errors='coerce')
    millidf = millidf.fillna(0)  # Fill NaN values with 0
    millidf = millidf.clip(lower=0)  # Ensure no negative values
    millidf = millidf.sort_values(by=['rackid', 'hostid'])
    millidf = millidf.reset_index(drop=True)
    print(f"Aggregated data shape: {millidf.shape}")
    
    label = 'test' if istest else 'train'
    label += f"_{len(rackids)}racks"
    millidf.to_csv(f"data/metadc_{label}.csv", index=False)
    
    rackids = sorted(list(rackids))
    print(f"{len(rackids)} racks")
    # print(f"Rack IDs: \n{rackids}")