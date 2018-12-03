import csv
import glob
import sys
import os
import xml.etree.ElementTree as ET
import os.path as osp

images_dir = '/home/adam/Pictures/vat'
annotations_dir = osp.join(images_dir, 'annotations')
annotations_csv = open('./vat_annotations.csv', 'w', newline='')
# classes_csv = open('./vat_classes.csv', 'w', newline='')
classes = ['code', 'number', 'date', 'check_code', 'buyer', 'seller', 'goods', 'amount_without_tax',
           'tax_rate', 'tax_amount', 'amount_with_tax', 'qrcode']

annotations_csv_writer = csv.writer(annotations_csv, dialect='excel')
classes_csv_writer = csv.writer(classes_csv)
for id, value in enumerate(classes):
    classes_csv_writer.writerow([value, id])

for image_file_path in glob.glob(images_dir + '/*.jpg'):
    images_dir, image_file = os.path.split(image_file_path)
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
        l = [image_file_path, str(bbox[0]), str(bbox[1]), str(bbox[2]), str(bbox[3]), cls]
        annotations_csv_writer.writerow(l)
