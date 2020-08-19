
file_name = '../reports2/Drug_119/hyper_para_tune_report_drug_119_features_200_hlayers_4_4_part0_new.txt'

with open(file_name, 'r') as f:
    l = ''
    for line in f:
        if '\t' not in line:
            l = line.split('\n')[0]
        if 'feat_200__4_layer_[200, 200, 180, 140, 120, 1]_epoch_100_drop_1.0_lr_0.1' in line:
            val = round(float(line.split('\t')[1]), 3)
            print(l + '\t' + str(val))