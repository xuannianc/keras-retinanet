from keras_retinanet.models import load_model
model = load_model('/path/to/model.h5', backbone_name='resnet50')