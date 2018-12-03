import os
import os.path as osp
import shutil
import glob
import random
import csv
import xml.etree.ElementTree as ET

images_dir = '/home/adam/Pictures/vat/train_seal_mask_removed'
annotations_dir = osp.join(images_dir, 'annotations_with_title')
classes = ['code', 'number', 'date', 'check_code', 'buyer', 'seller', 'goods', 'amount_without_tax',
           'tax_rate', 'tax_amount', 'amount_with_tax', 'qrcode', 'title']


def xml2csv(image_paths, csv_path):
    csv_file = open(csv_path, 'w', newline='')
    annotations_csv_writer = csv.writer(csv_file, dialect='excel')
    for image_path in image_paths:
        images_dir, image_file = os.path.split(image_path)
        image_file_name, image_file_ext = os.path.splitext(image_file)
        xml_file = image_file_name + '.xml'
        xml_file_path = os.path.join(annotations_dir, xml_file)
        if not os.path.exists(xml_file_path):
            continue
        tree = ET.parse(xml_file_path)
        root = tree.getroot()

        for obj in root.iter('object'):
            cls = obj.find('name').text
            assert cls in classes, cls + 'not in classes'
            xmlbox = obj.find('bndbox')
            bbox = (int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text), int(xmlbox.find('xmax').text),
                    int(xmlbox.find('ymax').text))
            l = [image_path, str(bbox[0]), str(bbox[1]), str(bbox[2]), str(bbox[3]), cls]
            annotations_csv_writer.writerow(l)


def split():
    special_image_paths = []
    common_image_paths = []
    for image_path in glob.glob(osp.join(images_dir, '*.jpg')):
        image_file = osp.split(image_path)[1]
        if image_file.startswith('0_'):
            common_image_paths.append(image_path)
        elif image_file.startswith('1_'):
            special_image_paths.append(image_path)
        else:
            image_file_parts = image_file.split('_')
            if len(image_file_parts) == 3:
                special_image_paths.append(image_path)
            elif len(image_file_parts) == 4:
                common_image_paths.append(image_path)
            else:
                print('[Warning] {} is neither common nor special.'.format(image_path))
    num_special_images = len(special_image_paths)
    num_common_images = len(common_image_paths)
    print('[INFO] num_special_images={}'.format(num_special_images))
    print('[INFO] num_common_images={}'.format(num_common_images))
    num_train_special_images = int(0.9 * num_special_images)
    num_train_common_images = int(0.9 * num_common_images)
    print('[INFO] num_train_special_images={}'.format(num_train_special_images))
    print('[INFO] num_train_common_images={}'.format(num_train_common_images))
    random.shuffle(special_image_paths)
    random.shuffle(common_image_paths)
    train_image_paths = special_image_paths[:num_train_special_images] + common_image_paths[:num_train_common_images]
    val_image_paths = special_image_paths[num_train_special_images:] + common_image_paths[num_train_common_images:]
    num_train_images = len(train_image_paths)
    num_val_images = len(val_image_paths)
    print('[INFO] num_train_images={}'.format(num_train_images))
    print('[INFO] num_val_images={}'.format(num_val_images))
    # generate csv
    xml2csv(train_image_paths, './train_annotations_{}_{}.csv'.format(num_train_images + num_val_images, num_train_images))
    xml2csv(val_image_paths, './val_annotations_{}_{}.csv'.format(num_train_images + num_val_images, num_val_images))


split()
