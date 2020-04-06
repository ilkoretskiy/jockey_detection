import io
import os
import PIL
import tensorflow as tf

from sklearn.model_selection import train_test_split

from object_detection.utils import dataset_util
import tools.collect_annotations as ca

flags = tf.app.flags
flags.DEFINE_string('annotations_dir', '', 'Path to annotations')
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
FLAGS = flags.FLAGS


def create_tf_example(example : ca.AnnotationInfo):
  
  with tf.io.gfile.GFile(example.image_path, 'rb') as fid:
    encoded_img = fid.read()

  encoded_img_io = io.BytesIO(encoded_img)
  image = PIL.Image.open(encoded_img_io)

  width = int(image.width)
  height = int(image.height)
  filename = example.image_filename # Filename of the image. Empty if image is not from file
  encoded_image_data = encoded_img # Encoded image bytes
  image_format = image.format # b'jpeg' or b'png'

  
  xmins = [] # List of normalized left x coordinates in bounding box (1 per box)
  xmaxs = [] # List of normalized right x coordinates in bounding box
             # (1 per box)
  ymins = [] # List of normalized top y coordinates in bounding box (1 per box)
  ymaxs = [] # List of normalized bottom y coordinates in bounding box
             # (1 per box)
  classes_text = [] # List of string class name of bounding box (1 per box)
  classes = [] # List of integer class id of bounding box (1 per box)

  jockey_class = 1
  jockey_class_name = "jockey"
  for annotation in example.annotations:
    xmin, ymin, xmax, ymax = annotation
    xmin = xmin / width
    xmax = xmax / width
    ymin = ymin / height
    ymax = ymax / height
    xmins.append(xmin)
    xmaxs.append(xmax)
    ymins.append(ymin)
    ymaxs.append(ymax)

    class_text = jockey_class_name
    classes_text.append(class_text.encode('utf8'))
    classes.append(jockey_class)


  tf_example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': dataset_util.int64_feature(height),
      'image/width': dataset_util.int64_feature(width),
      'image/filename': dataset_util.bytes_feature(filename.encode('utf8')),
      'image/source_id': dataset_util.bytes_feature(filename.encode('utf8')),
      'image/encoded': dataset_util.bytes_feature(encoded_image_data),
      'image/format': dataset_util.bytes_feature(image_format.encode('utf8')),
      'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
      'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
      'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
      'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
      'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
      'image/object/class/label': dataset_util.int64_list_feature(classes),
  }))
  return tf_example


def main(_):  
  all_annotations = ca.get_annotations(FLAGS.annotations_dir)
  datasets = train_test_split(all_annotations, test_size=0.2, random_state=0)
  names = ["jockey_train.record", "jockey_val.record"]
  os.makedirs(FLAGS.output_path, exist_ok=True)

  for dataset_annotations, output_name in zip(datasets, names):
    print ("Dataset : {}; {}".format(output_name, len(dataset_annotations)))
    full_output_path =  os.path.join(FLAGS.output_path, output_name)

    writer = tf.python_io.TFRecordWriter(full_output_path)
    for idx, annotation in enumerate(dataset_annotations):
      if idx % 10 == 0:
        print ("Process : #{} {} ".format(idx, annotation))
      tf_example = create_tf_example(annotation)
      writer.write(tf_example.SerializeToString())
    writer.close()


if __name__ == '__main__':
  tf.app.run()