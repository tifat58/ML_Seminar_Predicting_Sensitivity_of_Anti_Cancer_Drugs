import os, sys
from os.path import isfile, join, isdir
from os import listdir
import matplotlib

report_file_string = '../reports2/Drug_'

report_file_path = report_file_string + sys.argv[1]

file_list = [f for f in listdir(report_file_path) if isfile(join(report_file_path, f))]

for file in file_list:
    if 'hyper_para_tune_report' in file and '_new' in file:
        print(file)
        report_dict = dict()
        with open(os.path.join(report_file_path, file), 'r') as f:
            i = 0
            for line in f:
                if '\t' not in line:
                    key1 = line
                    report_dict.update({})
            print(i)