import os, sys
from os.path import isfile, join, isdir
from os import listdir

# Run to get the best models for a drug based on f1 or acc, the function runs through all the best model files and select the best
# run with two argument
# argv[1] = Drug Number e.g. 119, 131 etc
# argv[2] = f1 ; if to select best model based on f1 score
# argv[2] = acc ; if to select best model based on test accuracy

report_file_string = '../reports2/Drug_'

report_file_path = report_file_string + sys.argv[1]
if not isdir(report_file_path):
    print("Wrong Drug ID. Directroy doesn't exists.")
    exit(2)


f1 = 'F1 Score:'
acc = 'Test Accuracy:'
std = 'Test Accuracy STD:'
tr = 'Train Accuracy:'
hyper = sys.argv[3]
new = sys.argv[4]

best_score = 0


file_list = [f for f in listdir(report_file_path) if isfile(join(report_file_path, f))]

best_f = dict()
accs = dict()
stds = dict()
train = dict()


file_list = [f for f in listdir(report_file_path) if isfile(join(report_file_path, f))]
print(len(file_list))

best_value = 0
best_model = dict()

highest = []
counter = 0
for file in file_list:
    if 'hyper' not in file and "report" in file and "part" in file and hyper == "no":
        counter = counter +1
        #print(file)
        data_dict = dict()
        with open(os.path.join(report_file_path,file), 'r') as f:
            lines = f.readlines()
            i = 0
            while i < len(lines)-1:
                if acc in lines[i]:
                    i = i+1
                    while ":" not in lines[i] and i < len(lines)-1:
                        name, value1 = lines[i].split(u'\t')

                        value = value1.split('\n')[0]
                        #print(name, value)

                        i = i + 1

                        accs[str(name)] = float(value)

                i = i+1

            i = 0
            while i < len(lines)-1:

                if f1 in lines[i]:
                    i = i + 1
                    while ":" not in lines[i] and i < len(lines)-1:

                        name, value1 = lines[i].split(u'\t')
                        value = value1.split('\n')[0]

                        i = i + 1

                        tuple = (str(name), float(value))
                        highest.append(tuple)
                    else:
                        continue

                i = i+1

            i = 0
            while i < len(lines) - 1:

                if tr in lines[i]:
                    i = i + 1
                    while ":" not in lines[i] and i < len(lines) - 1:

                        name, value1 = lines[i].split(u'\t')
                        value = value1.split('\n')[0]

                        i = i + 1

                        train[str(name)] = float(value)
                    else:
                        continue

                i = i + 1

            i = 0
            while i < len(lines)-1:
                if std in lines[i]:
                    i = i+1
                    while ":" not in lines[i] and i < len(lines)-1:
                        name, value1 = lines[i].split(u'\t')

                        value = value1.split('\n')[0]

                        i = i + 1

                        stds[str(name)] = float(value)

                i = i+1

    elif "report" in file and "part" in file and hyper in file and new in file:
        #counter = counter + 1
        data_dict = dict()
        with open(os.path.join(report_file_path, file), 'r') as f:
            lines = f.readlines()
            i = 0
            while i < len(lines) - 1:
                if acc in lines[i]:
                    i = i + 1
                    while ":" not in lines[i] and i < len(lines) - 1:
                        name, value1 = lines[i].split(u'\t')

                        value = value1.split('\n')[0]
                        # print(name, value)

                        i = i + 1

                        accs[str(name)] = float(value)

                i = i + 1

            i = 0
            while i < len(lines) - 1:

                if f1 in lines[i]:
                    i = i + 1
                    while ":" not in lines[i] and i < len(lines) - 1:

                        name, value1 = lines[i].split(u'\t')
                        value = value1.split('\n')[0]

                        i = i + 1

                        tuple = (str(name), float(value))
                        highest.append(tuple)
                    else:
                        continue

                i = i + 1

            i = 0
            while i < len(lines) - 1:

                if tr in lines[i]:
                    i = i + 1
                    while ":" not in lines[i] and i < len(lines) - 1:

                        name, value1 = lines[i].split(u'\t')
                        value = value1.split('\n')[0]

                        i = i + 1

                        train[str(name)] = float(value)
                    else:
                        continue

                i = i + 1

            i = 0
            while i < len(lines) - 1:
                if std in lines[i]:
                    i = i + 1
                    while ":" not in lines[i] and i < len(lines) - 1:
                        name, value1 = lines[i].split(u'\t')

                        value = value1.split('\n')[0]

                        i = i + 1

                        stds[str(name)] = float(value)

                i = i + 1

print(counter)



sorted_highest = sorted(highest, key = lambda x: x[1])

print(sorted_highest[-10:])

highest_std = 0
lowest_std = 1
highest_acc = 0
best = ()

for val in sorted_highest[-10:]:
    print(val[0], str(round(val[1], 3)) + " & " + str(round(accs[val[0]], 3)) + " & " + str(round(stds[val[0]], 3)))
    if stds[val[0]] < lowest_std:
        lowest_std = stds[val[0]]

    if stds[val[0]] > highest_std:
        highest_std = stds[val[0]]

range = highest_std - lowest_std
print("lowest", lowest_std)
print("highest", highest_std)

thresh = float(sys.argv[2])
perc = (thresh * range) + lowest_std
print("threshold: ", perc)

print("best models")
print("name", "fscore", "test acc", "std", "train acc")

for val in sorted_highest[-10:]:
    if stds[val[0]] <= perc:
        print(val[0], val[1], accs[val[0]], stds[val[0]], train[val[0]])
        if highest_acc < accs[val[0]]:
            highest_acc = accs[val[0]]
            best = val[0], val[1], accs[val[0]], stds[val[0]], train[val[0]]


print("Best", best)


