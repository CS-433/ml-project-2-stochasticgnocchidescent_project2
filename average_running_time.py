import csv
import matplotlib

# reads data from a PAGE csv log file
def get_data_from_csv(file_name):
    rows = []
    reader = csv.reader(open(file_name, "r"), delimiter = ",")
    for row in reader:
        rows.append((float(row[0]), float(row[1]), float(row[2]), float(row[3])))
    return rows

# returns average of running time for epoch in a run of PAGE
# pass file name of the PAGE log file
def avg_rt(file_name):
    rows = get_data_from_csv(file_name)

    sum_diffs = 0
    for i in range(1, len(rows)):
        sum_diffs += rows[i][1] - rows[i-1][1]

    return sum_diffs/(len(rows)-1)