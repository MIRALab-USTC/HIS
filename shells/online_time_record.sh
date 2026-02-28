exec > "online_time_record.out" 2>&1
file_name=(hyp_online_test multiplier_online_test square_online_test \
des_perf_online_test ethernet_online_test wb_conmax_online_test)
i=0
# epsilon=0是全部为GNN，epsilon=1是全部为focal
while [ $i -le ${#file_name[@]} ]
do
    echo ${file_name[i]}
    python o8_online_time_record.py --file_name ${file_name[i]} --csv_save_file online_results_test_v4.csv
    let i++
done