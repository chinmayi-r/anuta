from glob import glob
import gzip
import json
import numpy as np
import pandas as pd
from tqdm import tqdm


agg_window = 10
agg_stride = agg_window//2

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
    
    millidata = []
    for trace in tqdm(traces):
        f = gzip.open(trace)
        record = json.loads(f.read())
        if not 'ingressBytes' in record: continue
        
        aggregates = {
            'server_hostname': record['server_hostname'],
            # 'sampling_freq_ms': record['sampling_freq'], 
            'aggregate_window': agg_window,
            'aggregate_stride': agg_stride,
            'ingressBytes_aggregate': aggregate_timeseries(record['ingressBytes'], agg_window, agg_stride, sum, True),
            # 'egressBytes_aggregate': aggregate_timeseries(record['egressBytes'], agg_window, agg_stride, sum, True),
            'inRxmitBytes_aggregate': aggregate_timeseries(record['inRxmitBytes'], agg_window, agg_stride, sum, True),
            # 'outRxmitBytes_aggregate': aggregate_timeseries(record['outRxmitBytes'], agg_window, agg_stride, sum, True),
            'inCongestionBytes_aggregate': aggregate_timeseries(record['inCongestionExperiencedBytes'], agg_window, agg_stride, sum, True),
            'connections_aggregate': aggregate_timeseries(record['connections'], agg_window, agg_stride, sum, True),
        }
        keys_to_zip = [k for k, v in aggregates.items() if isinstance(v, list)]
        flattened = [
            {**{k: aggregates[k] for k in ['server_hostname', 'aggregate_window', 'aggregate_stride']}, 
             **dict(zip(keys_to_zip, values))}
            for values in zip(*[aggregates[key] for key in keys_to_zip])
        ]
        
        record_len = len(record['ingressBytes'])
        ingressbytes = record['ingressBytes'][: record_len - (record_len % agg_window)]
        for i, aggregate in enumerate(flattened):
            start = i*agg_stride
            total_bytes = 0
            for j, val in enumerate(ingressbytes[start: start+agg_window]):
                key = f"ingressBytes_{j}"
                aggregate[key] = val
                total_bytes += val
            assert total_bytes == aggregate['ingressBytes_aggregate'], f"Aggregation {i}: {total_bytes=} ≠ {aggregate['ingressBytes_aggregate']}"
            millidata.append(aggregate)
    
    millidf = pd.DataFrame.from_dict(millidata)
    millidf.to_csv(f"data/meta_w{agg_window}_s{agg_stride}_all.csv", index=False)