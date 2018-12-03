import pprint
import os.path as osp


def find_missing_label_image_paths(csv_path, target_labels):
    csv_file = open(csv_path, 'r')
    statistics = {}
    lines = csv_file.readlines()
    for line in lines:
        line = line.strip()
        line_parts = line.split(',')
        image_path = line_parts[0]
        label = line_parts[-1]
        if statistics.get(image_path):
            statistics[image_path].append(label)
        else:
            statistics[image_path] = []
            statistics[image_path].append(label)
    missing_label_image_paths = {}
    for target_label in target_labels:
        missing_label_image_paths[target_label] = []
    for image_path in statistics:
        for target_label in target_labels:
            if target_label not in statistics[image_path]:
                missing_label_image_paths[target_label].append(image_path)
    return missing_label_image_paths
    # pprint.pprint(missing_label_image_paths)
    # print(len(missing_label_image_paths['check_code']))


def find_special_with_check_code(csv_path):
    csv_file = open(csv_path, 'r')
    statistics = {}
    lines = csv_file.readlines()
    for line in lines:
        line = line.strip()
        line_parts = line.split(',')
        image_path = line_parts[0]
        label = line_parts[-1]
        if statistics.get(image_path):
            statistics[image_path].append(label)
        else:
            statistics[image_path] = []
            statistics[image_path].append(label)
    for image_path in statistics:
        _, image_file = osp.split(image_path)
        if 'check_code' in statistics[image_path]:
            if len(image_file.split('_')) == 4 or image_file.startswith('0_'):
                continue
            else:
                print(image_path)


def find_common_without_check_code(csv_path):
    csv_file = open(csv_path, 'r')
    statistics = {}
    lines = csv_file.readlines()
    for line in lines:
        line = line.strip()
        line_parts = line.split(',')
        image_path = line_parts[0]
        label = line_parts[-1]
        if statistics.get(image_path):
            statistics[image_path].append(label)
        else:
            statistics[image_path] = []
            statistics[image_path].append(label)
    for image_path in statistics:
        _, image_file = osp.split(image_path)
        if 'check_code' not in statistics[image_path]:
            if len(image_file.split('_')) == 3 or image_file.startswith('1_'):
                continue
            else:
                print(image_path)


val_csv_path = 'val_annotations_1090_110.csv'
train_csv_path = 'train_annotations_1090_980.csv'
labels = ['code', 'number', 'date', 'buyer', 'seller', 'goods', 'amount_without_tax',
          'tax_rate', 'tax_amount', 'amount_with_tax', 'qrcode', 'title']
val_missings = find_missing_label_image_paths(val_csv_path, labels)
train_missings = find_missing_label_image_paths(train_csv_path, labels)
pprint.pprint(train_missings)
pprint.pprint(val_missings)
find_special_with_check_code(val_csv_path)
find_special_with_check_code(train_csv_path)
find_common_without_check_code(train_csv_path)
find_common_without_check_code(val_csv_path)
