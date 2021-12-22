import csv
import matplotlib
import numpy as np

# useful functions for reading logs


# reads data from csv containing log produced by run of SARAH
# columns are
# ~ epoch number, number of computed gradients so far, current training loss, current test accuracy
def get_data_from_csv_sarah(file_name):
    rows = []
    reader = csv.reader(open(file_name, "r"), delimiter = ",")
    for row in reader:
        rows.append((float(row[0]), int(row[1]), float(row[2]), float(row[3])))
    return rows

# reads data from csv containing log produced by run of PAGE
# columns are
# iteration number, elapsed seconds, computed gradients so far, current training loss, current test accuracy
def get_data_from_csv(file_name):
    rows = []
    reader = csv.reader(open(file_name, "r"), delimiter = ",")
    for row in reader:
        rows.append((int(row[0]), float(row[1]), int(row[2]), float(row[3]), float(row[4])))
    return rows

# given a folder path containing files 0_common_suffix, ... (num_experiments-1)_common_suffix
# return three lists containing for each log row the average over experiments of gradient count, training loss, test accuracy
# same for std over experiments
# works for ProxSARAH log files
def avg_from_files_sarah(folder, common_suffix, num_experiments):
    avg_train_loss = []
    avg_test_acc = []
    avg_grad_count = []
    std_loss = []
    std_acc = []
    
    for i in range(num_experiments):
        rows = get_data_from_csv_sarah(folder+"/"+str(i)+ "_"+ common_suffix)
        for j, r in enumerate(rows):
            if i == 0:
                avg_grad_count.append([r[1]])
                avg_train_loss.append([r[2]])
                avg_test_acc.append([r[3]])
                std_loss.append([r[2]])
                std_acc.append([r[3]])
            else:
                avg_grad_count[j].append(r[1])
                avg_train_loss[j].append(r[2])
                avg_test_acc[j].append(r[3])
                std_loss[j].append(r[2])
                std_acc[j].append(r[3])
    
    num_events = len(avg_grad_count)

    for j in range(num_events):
        sum_grad = 0
        sum_loss = 0
        sum_acc = 0
        std_loss[j] = np.std(np.array(std_loss[j]))
        std_acc[j] = np.std(np.array(std_acc[j]))
        for i in range(num_experiments):
            sum_grad += avg_grad_count[j][i]
            sum_loss += avg_train_loss[j][i]
            sum_acc += avg_test_acc[j][i]
        avg_grad_count[j] = int(sum_grad/num_experiments)
        avg_train_loss[j] = sum_loss/num_experiments
        avg_test_acc[j] = sum_acc/num_experiments

    return avg_grad_count, avg_train_loss, avg_test_acc, std_loss, std_acc

# given a folder path containing files 0_common_suffix, ... (num_experiments-1)_common_suffix
# return three lists containing for each log row the average over experiments of gradient count, training loss, test accuracy
# same for std over experiments
# works for PAGE log files
def avg_from_files(folder, common_suffix, num_experiments):
    avg_train_loss = []
    avg_test_acc = []
    avg_grad_count = []
    std_loss = []
    std_acc = []
    
    for i in range(num_experiments):
        rows = get_data_from_csv(folder + "/" + str(i)+ "_"+ common_suffix)
        for j, r in enumerate(rows):
            if j >= len(avg_grad_count):
                avg_grad_count.append([r[2]])
                avg_train_loss.append([r[3]])
                std_loss.append([r[3]])
                avg_test_acc.append([r[4]])
                std_acc.append([r[4]])
                
            else:
                avg_grad_count[j].append(r[2])
                avg_train_loss[j].append(r[3])
                std_loss[j].append(r[3])
                avg_test_acc[j].append(r[4])
                std_acc[j].append(r[4])
    
    num_events = len(avg_grad_count)

    for j in range(num_events):
        sum_grad = 0
        sum_loss = 0
        sum_acc = 0
        std_loss[j] = np.std(np.array(std_loss[j]))
        std_acc[j] = np.std(np.array(std_acc[j]))
        for g in avg_grad_count[j]:
            sum_grad += g
        for l in avg_train_loss[j]:
            sum_loss += l
        for a in avg_test_acc[j]:
            sum_acc += a
        avg_grad_count[j] = int(sum_grad/len(avg_grad_count[j]))
        avg_train_loss[j] = sum_loss/len(avg_train_loss[j])
        avg_test_acc[j] = sum_acc/len(avg_test_acc[j])

    return avg_grad_count, avg_train_loss, avg_test_acc, std_loss, std_acc
