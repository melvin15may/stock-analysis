import dataCollection as dc
import recognizer as rz
import genetic_algo as ga
import pandas as pd
import sys
import glob
import os
import numpy as np
from datetime import datetime
import json
import multiprocessing

# window size
WLEN = 29


def main():
    # 2 different states; 1: Data collection state and 2: data processing(pattern
    # recognition) state
    state = int(sys.argv[1])
    directory_name = 'data'
    try:
        directory_name = sys.argv[2]
    except IndexError:
        pass

    if state == 1:
        dc.get_data()
    elif state == 2:
        file_names = []
        for filename in glob.glob(os.path.join(directory_name, '*.csv')):
            file_names.append(filename)
        file_names = file_names[:2]
        with multiprocessing.Pool(processes=2) as pp:
            results = pp.map(run_GA, file_names)
            # run_GA(file_name)
        json_data = {}
        for ind, f in enumerate(file_names):
            json_data[os.path.basename(filename).split('.')[0]] = results[ind]
        with open("data/GA.txt", "w") as jf:
            json.dump(json_data, jf)

    else:
        data = ["symbol,pipoutput,pipoutputtime\n"]
        for filename in glob.glob(os.path.join(directory_name, '*.csv')):
            df = dc.read_csv(filename, chunksize=WLEN)
            pattern, pattern_time = get_resistance_support(df)
            for p in range(0, len(pattern)):
                data.append(
                    ",".join([os.path.basename(filename).split('.')[0], ";".join(map(str, pattern[p])), ";".join(pattern_time[p])]) + "\n")

        dc.write_csv(data=data)


# Run GA on past data
def run_GA(file_name):
    json_data = {}
    df = dc.read_csv(file_name, chunksize=WLEN)
    prices = []
    time = []
    for data in df:
        prices += list(map(float, data['close'].values.tolist()))
        time += data['timestamp'].values.tolist()
    prices = normalized(np.array(prices))[0]
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), " Run GA for ", file_name)
    wlength = ga.runGA(list(range(0, len(time))), prices)
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), " End GA", wlength)
    return wlength
    """
    with open('data/GA.txt', 'w') as jf:
        json.dump(json_data, jf)
    """


# Normalize
def normalized(a, axis=0, order=1):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis)


def get_resistance_support(data_file):
    chunk_left_P = []
    chunk_left_P_time = []
    detected_patterns = {}
    detected_patterns_time = {}
    for data in data_file:
        start_row = 0
        P = chunk_left_P + list(map(float, data['close'].values.tolist()))
        P_time = chunk_left_P_time + data['timestamp'].values.tolist()
        P_len = len(P)

        # Remove patterns which have already been triggered
        # when current price is higher that the breakout pattern
        P_max = max(P)
        for k, v in list(detected_patterns.items()):
            if k <= P_max:
                del detected_patterns[k]
                del detected_patterns_time[k]

        # Remove patterns which are void
        # when the current price is lower than the support
        P_min = min(P)
        for k, v in list(detected_patterns.items()):
            if min(v) > P_min:
                del detected_patterns[k]
                del detected_patterns_time[k]

        while P_len >= (start_row + WLEN):
            # print(P_len,start_row,start_row + WLEN)
            SP, SP_time = rz.PIP_identification(
                P[start_row:start_row + WLEN], P_time[start_row:start_row + WLEN])
            # print(SP,SP_time)
            if rz.inverse_head_and_shoulder_rule(SP):
                detected_patterns[max(SP)] = SP
                detected_patterns_time[max(SP)] = SP_time
                start_row += (WLEN - 1)
            start_row += 1
        chunk_left_P = P[start_row:]
        chunk_left_P_time = P_time[start_row:]
    return list(detected_patterns.values()), list(detected_patterns_time.values())


if __name__ == '__main__':
    main()
