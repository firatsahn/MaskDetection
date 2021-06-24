import cv2
import numpy as np
import tensorflow as tf

from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

#print(tf.__version__)


labelmap_path = 'training/labelmap.pbtxt'
category_index = label_map_util.create_category_index_from_labelmap(labelmap_path, use_display_name=True)
tf.keras.backend.clear_session()

model = tf.saved_model.load('inference_graph/saved_model/')

def load_image_into_numpy_array(path):

    img_data = cv2.imread(path)
    #img = np.array(img_data)
    img = np.array(img_data).astype(np.uint8)
    #img = np.reshape(img,(512,512,3)).astype(np.uint8)
    return img

def run_inference_for_single_image(model, image):
  image = np.asarray(image)
  # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
  input_tensor = tf.convert_to_tensor(image)
  # The model expects a batch of images, so add an axis with `tf.newaxis`.
  input_tensor = input_tensor[tf.newaxis,...]

  # Run inference
  model_fn = model.signatures['serving_default']
  output_dict = model_fn(input_tensor)

  # Tüm çıktılar batch tensörlerdir
  # Convert to numpy arrays, and take index [0] to remove the batch dimension.
  # We're only interested in the first num_detections.
  num_detections = int(output_dict.pop('num_detections'))
  
  output_dict = {key:value[0, :num_detections].numpy() 
                 for key,value in output_dict.items()}
  
  output_dict['num_detections'] = num_detections
  
  # detection_classes should be ints.
  output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)
    
  # Handle models with masks:
  if 'detection_masks' in output_dict:
    print("----------------------------*/**/*/")
    # Reframe the the bbox mask to the image size.
    detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
              output_dict['detection_masks'], output_dict['detection_boxes'],
               image.shape[0], image.shape[1])
    print(detection_masks_reframed)
    detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5,
                                       tf.uint8)
    output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()

  return output_dict

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Kamera Açılırken Bir Hata ile Karşılaşıldı!")
    exit()
while True:
    ret,frame = cap.read()
    
    output_dict = run_inference_for_single_image(model, frame)
    
    vis_util.visualize_boxes_and_labels_on_image_array(
      frame,
      output_dict['detection_boxes'],
      output_dict['detection_classes'],
      output_dict['detection_scores'],
      category_index,
      instance_masks=output_dict.get('detection_masks_reframed', None),
      use_normalized_coordinates=True,
      line_thickness=2,
      min_score_thresh=0.70)
    
    
    
    cv2.imshow('frame',frame)
    key = cv2.waitKey(1) & 0xFF
    if(key == ord('q')):
        break


cap.release()
cv2.destroyAllWindows()

