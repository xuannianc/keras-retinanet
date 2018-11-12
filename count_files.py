import csv

vat_filenames = set()
train_csv_filename = 'train_annotations.csv'
val_csv_filename = 'val_annotations.csv'
for csv_filename in [train_csv_filename, val_csv_filename]:
    for line in csv.reader(open(csv_filename)):
        vat_filename = line[0].split('/')[-1]
        vat_filenames.add(vat_filename)
    print(len(vat_filenames))
    vat_filenames.clear()
