import os
import numpy as np
import csv
import argparse
from collections import defaultdict

def process_npy_files(directory, file_name, csv_save_file):
    file_groups = defaultdict(list)
    
    # Group files by the part of the filename before "seed"
    for filename in os.listdir(directory):
        if filename.endswith(".npy"):
            key = filename.split("seed")[0]
            file_groups[key].append(filename)
            if 'CMO' in filename:
                filepath = os.path.join(directory, filename)
                data = np.load(filepath, allow_pickle=True).item()
                if 'sel_percent_1.0' in data.keys():
                    default_nd_degradation = data['sel_percent_1.0']['and'][0] - data['sel_percent_1.0']['and'][1]
                else:
                    default_nd_degradation = 0
                print('default_nd_degradation is', default_nd_degradation)
            else:
                default_nd_degradation = 0
    csv_filename = f'./csv_results/{csv_save_file}'
    # if not os.path.exists(csv_filename):
    #     os.makedirs(csv_filename)
    # Create a CSV file to write the results
    with open(csv_filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        # Write the header
        writer.writerow(['Filename', 'Pure Inference Mean', 'Pure Inference Std', 
                         'Offline Mean', 'Offline Std', 'Online Mean', 'Online Std', 
                         'Nd Degradation Mean', 'Nd Degradation Std', 'Init Nd Mean', 'Default Nd Degradatoin', 'Nd Degradation rate'])
        for key, filenames in file_groups.items():
            print('the key is', key)
            pure_inference_times = defaultdict(list)
            offline_times = defaultdict(list)
            online_times = defaultdict(list)
            nd_degradations = defaultdict(list)
            init_nds = defaultdict(list)
            for filename in filenames:
                filepath = os.path.join(directory, filename)
                try:
                    data = np.load(filepath, allow_pickle=True).item()
                    for k in data.keys():
                        pure_inference_time = data[k]['time_policy_inference'][0]
                        # data[k]['offline_time_all'][0] - data[k]['time_process_csv'][0]
                        offline_time_all = data[k]['offline_time_all'][0]
                        online_time_all = data[k]['online_time_all'][0]
                        nd_degradation = data[k]['and'][0] - data[k]['and'][1]
                        init_nd = data[k]['and'][0]
                        if 'sel_percent' in k:
                            pure_inference_times[k].append(pure_inference_time)
                            offline_times[k].append(offline_time_all)
                            online_times[k].append(online_time_all)
                            nd_degradations[k].append(nd_degradation)
                            init_nds[k].append(init_nd)
                        print(f" File: {filename}")
                        print(f" pure_inference_time of {k}: {pure_inference_time}")
                        print(f" offline_time_all of {k}: {offline_time_all}")
                        print(f" online_time_all of {k}: {online_time_all}")
                        print(f" nd_degradation of {k}: {nd_degradation}")
                        print(f" init_nd of {k}: {init_nd}")
                except Exception as e:
                    print(f"Could not process file {filename}: {e}")
            print(f'The method is {key}')
            for subkey in offline_times.keys():
                print(f'The {subkey} results')
                if pure_inference_times[subkey]:
                    pure_inference_mean = np.mean(pure_inference_times[subkey])
                    pure_inference_std = np.std(pure_inference_times[subkey])
                    print(f"{subkey} - Pure inference times - Mean: {pure_inference_mean}, Std: {pure_inference_std}")
                if offline_times[subkey]:
                    offline_mean = np.mean(offline_times[subkey])
                    offline_std = np.std(offline_times[subkey])
                    print(f"{subkey} - Offline times all - Mean: {offline_mean}, Std: {offline_std}")
                
                if online_times[subkey]:
                    online_mean = np.mean(online_times[subkey])
                    online_std = np.std(online_times[subkey])
                    print(f"{subkey} - Online times all - Mean: {online_mean}, Std: {online_std}")

                if nd_degradations[subkey]:
                    nd_degradation_mean = np.mean(nd_degradations[subkey])
                    nd_degradation_std = np.std(nd_degradations[subkey])
                    print(f"{subkey} - Nd degradation - Mean: {nd_degradation_mean}, Std: {nd_degradation_std}")

                if init_nds[subkey]:
                    init_nd_mean = np.mean(init_nds[subkey])
                    init_nd_std = np.std(init_nds[subkey])
                    print(f"{subkey} - init_nd - Mean: {init_nd_mean}, Std: {init_nd_std}")

                # Write the results to the CSV file
                nd_degradation_rate = (default_nd_degradation - nd_degradation_mean)/init_nd_mean * 100
                writer.writerow([file_name +'_'+ key + subkey, pure_inference_mean, pure_inference_std, 
                                 offline_mean, offline_std, online_mean, online_std, 
                                 nd_degradation_mean, nd_degradation_std, init_nd_mean, default_nd_degradation, nd_degradation_rate])
            
            # # Write a blank row to separate different filenames
            # writer.writerow([])


# Example usage:
# process_npy_files('path/to/your/directory')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process online file.")
    parser.add_argument('--file_name', type=str, default='hyp_online_test', help='Directory containing .npy files')
    parser.add_argument('--csv_save_file', type=str, default='online_results_test.csv')
    args = parser.parse_args()
    directory = './online_test/mfs2'
    subdirectory_path = os.path.join(directory, f'{args.file_name}')
    process_npy_files(subdirectory_path, args.file_name, args.csv_save_file)
