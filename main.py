import dataCollection as dc
import recognizer as rz
import genetic_algo as ga
import template_patterns as tp
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

TEMPLATES_DICT = tp.template_patterns()

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
        # file_names = file_names[:2]
        with multiprocessing.Pool(processes=3) as pp:
            results = pp.map(run_GA, file_names)
            # run_GA(file_name)
        json_data = {}
        for ind, f in enumerate(file_names):
            json_data[os.path.basename(f).split('.')[0]] = results[ind]
        with open("data/GA.txt", "w") as jf:
            json.dump(json_data, jf)

    else:
        """
        data = ["symbol,pipoutput,pipoutputtime\n"]
        for filename in glob.glob(os.path.join(directory_name, '*.csv')):
            df = dc.read_csv(filename, chunksize=WLEN)
            pattern, pattern_time = get_resistance_support(df)
            for p in range(0, len(pattern)):
                data.append(",".join([os.path.basename(filename).split('.')[0], ";".join(
                    map(str, pattern[p])), ";".join(pattern_time[p])]) + "\n")

        dc.write_csv(data=data)
        """
        json_data = json.load(open("data/GA.txt"))
        arguments = []
        data = ["symbol,pipoutput,pipoutputtime\n"]
        for key in json_data:
            arguments.append((key, sorted(json_data[key]),directory_name))
        with multiprocessing.Pool(processes=3) as pp:
            results = pp.starmap(pattern_recog,arguments)
        for r in results:
            data += r
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
    wlength = []
    if len(prices) > 0:
        print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), " Run GA", file_name)
        wlength = ga.runGA(list(range(0, len(time))), prices)
        print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
              " End GA", file_name, wlength)
    return wlength


# Normalize
def normalized(a, axis=0, order=1):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis)


def pattern_recog(key, windows, directory_name):
    #json_data = json.load(open(json_file))
    test_files = list(glob.glob(os.path.join(directory_name, '*.csv')))
    data = []
    print("Pattern recognition for", key)
    if not os.path.join(directory_name, (key + '.csv')) in test_files:
        print("Downloading data for ", key)
        dc.downloadHistory_stocks(key, directory_name=directory_name)
    prices = []
    times = []
    try:
        df = dc.read_csv(os.path.join(
            directory_name, key + '.csv'), chunksize=WLEN)
        for d in df:
            prices += list(map(float, d['close'].values.tolist()))
            times += d['timestamp'].values.tolist()
        pattern, pattern_time = get_resistance_support(prices, times, windows)
        for p in range(0, len(pattern)):
            data.append(",".join(
                [key, ";".join(map(str, pattern[p])), ";".join(pattern_time[p])]) + "\n")
    except FileNotFoundError:
        print("File not found error", key)
        pass
    return data


def get_resistance_support(prices, times, windows, recog_type="rule"):
    detected_patterns = {}
    detected_patterns_time = {}
    windows = [0] + windows + [len(prices)]
    for ind in range(1, len(windows)):
        # Remove patterns which have already been triggered
        # when current price is higher that the breakout pattern
        if prices[windows[ind]:]:
            P_max = max(prices[windows[ind]:])
            for k, v in list(detected_patterns.items()):
                if k <= P_max:
                    del detected_patterns[k]
                    del detected_patterns_time[k]

            # Remove patterns which are void
            # when the current price is lower than the support
            P_min = min(prices[windows[ind]:])
            for k, v in list(detected_patterns.items()):
                if min(v) > P_min:
                    del detected_patterns[k]
                    del detected_patterns_time[k]

        if (windows[ind] - windows[ind - 1]) >= 7:
            # print(P_len,start_row,start_row + WLEN)
            SP, SP_time, SP_indexes = rz.PIP_identification(prices[windows[ind - 1]:windows[ind]], times[windows[ind - 1]:windows[ind]])
            if recog_type=="rule":
                if rz.inverse_head_and_shoulder_rule(SP):
                    detected_patterns[max(SP)] = SP
                    detected_patterns_time[max(SP)] = SP_time
                else:
                    distortion = rz.template_matching(np.array(SP),np.array(SP_indexes), TEMPLATES_DICT['inverse head and shoulders']['y'], TEMPLATES_DICT['inverse head and shoulders']['x'])
                    #print(distortion)
                    #if rz.inverse_head_and_shoulder_rule(SP):
                    if distortion < 0.12:
                        detected_patterns[max(SP)] = SP
                        detected_patterns_time[max(SP)] = SP_time
    return list(detected_patterns.values()), list(detected_patterns_time.values())


if __name__ == '__main__':
    main()
