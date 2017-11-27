

#--------------------------------------------------------------------------------------------------------------------
# Configure environment:

https://www.tensorflow.org/install/install_mac
pip install tensorflow
pip install opencv-python
pip install shapely
pip install google-cloud



#--------------------------------------------------------------------------------------------------------------------
# Train Model

to label images use https://github.com/tzutalin/labelImg

python xml_to_csv.py

python generate_tfrecord.py --csv_input=data/train_labels.csv --output_path=data/train.record
python generate_tfrecord.py --csv_input=data/test_labels.csv --output_path=data/test.record

# -------------------------------
# Retrain

# Upload nessesary data
# data
scp -i "Documents/keys/odats.pem" -r /Users/odats/Documents/Projecets/elements-ai/make_object_detection_model/data  ubuntu@54.171.76.250:models/object_detection
#config
scp -i "Documents/keys/odats.pem" -r /Users/odats/Documents/Projecets/elements-ai/make_object_detection_model/training  ubuntu@54.171.76.250:models/object_detection
# models
scp -i "Documents/keys/odats.pem" -r /Users/odats/Documents/Projecets/oelements-ai/make_object_detection_model/models/faster_rcnn_resnet101_coco_11_06_2017 ubuntu@54.171.76.250:models/object_detection/models


export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim

sudo python setup.py install

#Train
rm -rf elements_inference_graph
rm -rf training_model

python train.py --logtostderr --train_dir=training_model/ --pipeline_config_path=training/elements_ec2.config

#Exports
python export_inference_graph.py \
    --input_type image_tensor \
    --pipeline_config_path training/elements_ec2.config \
    --trained_checkpoint_prefix training_model/model.ckpt-500 \
    --output_directory elements_inference_graph

# Copy
scp -i "Documents/keys/odats.pem" -r ubuntu@54.171.76.250:models/object_detection/elements_inference_graph/frozen_inference_graph.pb /Users/odats/Documents/Projecets/elements_inference_graph/frozen_inference_graph.pb

scp -i "Documents/keys/odats.pem" -r ubuntu@54.171.76.250:models/object_detection/elements_inference_graph /Users/odats/Documents/Projecets/elements_inference_graph
