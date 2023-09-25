import csv
import config

def add_report_data(data):
    with open(config.report_file_path, 'a', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        print(data)
        csvwriter.writerow(data)