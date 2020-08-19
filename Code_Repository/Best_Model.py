import os, sys
from os.path import isfile, join, isdir
from os import listdir

# Run to get the best model for a drug, the function runs through all the best model files and select the best
# run with two argument
# argv[1] = Drug Number e.g. 119, 131 etc
# argv[2] = fs ; if to select best model based on f1 score
# argv[2] = fw ; if to select best model based on f1 weighted score

report_file_string = '../reports2/Drug_'

report_file_path = report_file_string + sys.argv[1]
if not isdir(report_file_path):
    print("Wrong Drug ID. Directroy doesn't exists.")
    exit(2)

if sys.argv[2] == 'fs':
    criteria = 'For F1 Score:'
else:
    criteria = 'for F1 Weighted Score:'

file_list = [f for f in listdir(report_file_path) if isfile(join(report_file_path, f))]


best_value = 0
best_model = dict()
for file in file_list:
    check = False

    if "Best_model" in file and "part" in file:
        # print(file)
        data_dict = dict()
        with open(os.path.join(report_file_path,file), 'r') as f:
            for line in f:
                if criteria in line:
                    check = True
                if line == '\n':
                    check = False
                if check:
                    if '\t' in line:
                        name, value1 = line.split(u'\t')
                        value = value1.split('\n')[0]
                        data_dict.update({name:float(value)})

                        # if name == 'F1 Score Weighted:':
                        #     if float(value) > best_value:
                        #         best_value = float(value)
                        #         best_model = data_dict.copy()
                    elif 'feat' in line:
                        value = line.split('\n')[0]
                        data_dict.update({'model': value})

        # print('\n')
        # print(data_dict)
        if 'F1 Score Weighted:' in data_dict.keys() and sys.argv[2] != 'fs':
            if data_dict['F1 Score Weighted:'] > best_value:
                best_value = data_dict['F1 Score Weighted:']
                best_model = data_dict.copy()

        if 'F1 Score:' in data_dict.keys() and sys.argv[2] == 'fs':
            if data_dict['F1 Score:'] > best_value:
                best_value = data_dict['F1 Score:']
                best_model = data_dict.copy()

print("\nBest Model for Drug:" + sys.argv[1] + ' ' + criteria + '\n')
for k,v in best_model.items():
    print(k,v)