# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import os
import cv2
from typing import Union, List

import numpy as np
import torch
import onnxruntime
import os, sys
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
sys.path.insert(0, repo_root)
from wan.utils.utils import try_load_image
from pose2d_utils import (
    read_img,
    box_convert_simple,
    bbox_from_detector,
    crop,
    keypoints_from_heatmaps,
    load_pose_metas_from_kp2ds_seq
)


class SimpleOnnxInference(object):
    def __init__(self, checkpoint, device='cuda', reverse_input=False, **kwargs):
        if isinstance(device, str):
            device = torch.device(device)
        if device.type == 'cuda':
            device = '{}:{}'.format(device.type, device.index)
            providers = [("CUDAExecutionProvider", {"device_id": device[-1:] if device[-1] in [str(_i) for _i in range(10)] else "0"}), "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]
        self.device = device
        if not os.path.exists(checkpoint):
            raise RuntimeError("{} is not existed!".format(checkpoint))
        
        if os.path.isdir(checkpoint):
            checkpoint = os.path.join(checkpoint, 'end2end.onnx')

        self.session = onnxruntime.InferenceSession(checkpoint,
                                                    providers=providers
                                                    )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.input_resolution = self.session.get_inputs()[0].shape[2:] if not reverse_input else self.session.get_inputs()[0].shape[2:][::-1]
        self.input_resolution = np.array(self.input_resolution)
        

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
    

    def get_output_names(self):
        output_names = []
        for node in self.session.get_outputs():
            output_names.append(node.name)
        return output_names


    def set_device(self, device):
        if isinstance(device, str):
            device = torch.device(device)
        if device.type == 'cuda':
            device = '{}:{}'.format(device.type, device.index)
            providers = [("CUDAExecutionProvider", {"device_id": device[-1:] if device[-1] in [str(_i) for _i in range(10)] else "0"}), "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]
        self.session.set_providers(providers)
        self.device = device


class Yolo(SimpleOnnxInference):
    def __init__(self, checkpoint, device='cuda', threshold_conf=0.05, threshold_multi_persons=0.1, input_resolution=(640, 640), threshold_iou=0.5, threshold_bbox_shape_ratio=0.4, cat_id=[1], select_type='max', strict=True, sorted_func=None, **kwargs):
        super(Yolo, self).__init__(checkpoint, device=device, **kwargs)
        
        model_inputs = self.session.get_inputs()
        input_shape = model_inputs[0].shape

        self.input_width = 640
        self.input_height = 640
        
        self.threshold_multi_persons = threshold_multi_persons
        self.threshold_conf = threshold_conf
        self.threshold_iou = threshold_iou
        self.threshold_bbox_shape_ratio = threshold_bbox_shape_ratio
        self.input_resolution = input_resolution
        self.cat_id = cat_id
        self.select_type = select_type
        self.strict = strict
        self.sorted_func = sorted_func
        
        
    def preprocess(self, input_image):
        """
        Preprocesses the input image before performing inference.

        Returns:
            image_data: Preprocessed image data ready for inference.
        """
        img = read_img(input_image)
        # Get the height and width of the input image
        img_height, img_width = img.shape[:2]
        # Resize the image to match the input shape
        img = cv2.resize(img, (self.input_resolution[1], self.input_resolution[0]))
        # Normalize the image data by dividing it by 255.0
        image_data = np.array(img) / 255.0
        # Transpose the image to have the channel dimension as the first dimension
        image_data = np.transpose(image_data, (2, 0, 1))  # Channel first
        # Expand the dimensions of the image data to match the expected input shape
        # image_data = np.expand_dims(image_data, axis=0).astype(np.float32)
        image_data = image_data.astype(np.float32)
        # Return the preprocessed image data
        return image_data, np.array([img_height, img_width])

    
    def postprocess(self, output, shape_raw, cat_id=[1]):
        """
        Performs post-processing on the model's output to extract bounding boxes, scores, and class IDs.

        Args:
            input_image (numpy.ndarray): The input image.
            output (numpy.ndarray): The output of the model.

        Returns:
            numpy.ndarray: The input image with detections drawn on it.
        """
        # Transpose and squeeze the output to match the expected shape

        outputs = np.squeeze(output)
        if len(outputs.shape) == 1:
            outputs = outputs[None]
        if output.shape[-1] != 6 and output.shape[1] == 84:
            outputs = np.transpose(outputs)
        
        # Get the number of rows in the outputs array
        rows = outputs.shape[0]

        # Calculate the scaling factors for the bounding box coordinates
        x_factor = shape_raw[1] / self.input_width
        y_factor = shape_raw[0] / self.input_height

        # Lists to store the bounding boxes, scores, and class IDs of the detections
        boxes = []
        scores = []
        class_ids = []

        if outputs.shape[-1] == 6:
            max_scores = outputs[:, 4]
            classid = outputs[:, -1]
            
            threshold_conf_masks = max_scores >= self.threshold_conf
            classid_masks = classid[threshold_conf_masks] != 3.14159

            max_scores = max_scores[threshold_conf_masks][classid_masks]
            classid = classid[threshold_conf_masks][classid_masks]

            boxes = outputs[:, :4][threshold_conf_masks][classid_masks]
            boxes[:, [0, 2]] *= x_factor
            boxes[:, [1, 3]] *= y_factor
            boxes[:, 2] = boxes[:, 2] - boxes[:, 0]
            boxes[:, 3] = boxes[:, 3] - boxes[:, 1]
            boxes = boxes.astype(np.int32)

        else:
            classes_scores = outputs[:, 4:]
            max_scores = np.amax(classes_scores, -1)
            threshold_conf_masks = max_scores >= self.threshold_conf

            classid = np.argmax(classes_scores[threshold_conf_masks], -1)

            classid_masks = classid!=3.14159
            
            classes_scores = classes_scores[threshold_conf_masks][classid_masks]
            max_scores = max_scores[threshold_conf_masks][classid_masks]
            classid = classid[classid_masks]
    
            xywh = outputs[:, :4][threshold_conf_masks][classid_masks]

            x = xywh[:, 0:1]
            y = xywh[:, 1:2]
            w = xywh[:, 2:3]
            h = xywh[:, 3:4]
    
            left = ((x - w / 2) * x_factor)
            top = ((y - h / 2) * y_factor)
            width = (w * x_factor)
            height = (h * y_factor)
            boxes = np.concatenate([left, top, width, height], axis=-1).astype(np.int32)

        boxes = boxes.tolist()
        scores = max_scores.tolist()
        class_ids = classid.tolist()

        # Apply non-maximum suppression to filter out overlapping bounding boxes
        indices = cv2.dnn.NMSBoxes(boxes, scores, self.threshold_conf, self.threshold_iou)
        # Iterate over the selected indices after non-maximum suppression
        
        results = []
        for i in indices:
            # Get the box, score, and class ID corresponding to the index
            box = box_convert_simple(boxes[i], 'xywh2xyxy')
            score = scores[i]
            class_id = class_ids[i]
            results.append(box + [score] + [class_id])
            # # Draw the detection on the input image

        # Return the modified input image
        return np.array(results)

    
    def process_results(self, results, shape_raw, cat_id=[1], single_person=True):
        if isinstance(results, tuple):
            det_results = results[0]
        else:
            det_results = results

        person_results = []
        person_count = 0
        if len(results):
            max_idx = -1
            max_bbox_size = shape_raw[0] * shape_raw[1] * -10
            max_bbox_shape = -1
            
            bboxes = []
            idx_list = []
            for i in range(results.shape[0]):
                bbox = results[i]
                if (bbox[-1] + 1 in cat_id) and (bbox[-2] > self.threshold_conf):
                    idx_list.append(i)
                    bbox_shape = max((bbox[2] - bbox[0]), ((bbox[3] - bbox[1])))
                    if bbox_shape > max_bbox_shape:
                        max_bbox_shape = bbox_shape
            
            results = results[idx_list]

            for i in range(results.shape[0]):
                bbox = results[i]
                bboxes.append(bbox)
                if self.select_type == 'max':
                    bbox_size = (bbox[2] - bbox[0]) * ((bbox[3] - bbox[1]))
                elif self.select_type == 'center':
                    bbox_size = (abs((bbox[2] + bbox[0]) / 2 - shape_raw[1]/2)) * -1
                bbox_shape = max((bbox[2] - bbox[0]), ((bbox[3] - bbox[1])))
                if bbox_size > max_bbox_size:
                    if (self.strict or max_idx != -1) and bbox_shape < max_bbox_shape * self.threshold_bbox_shape_ratio:
                        continue
                    max_bbox_size = bbox_size
                    max_bbox_shape = bbox_shape
                    max_idx = i

            if self.sorted_func is not None and len(bboxes) > 0:
                max_idx = self.sorted_func(bboxes, shape_raw)
                bbox = bboxes[max_idx]
                if self.select_type == 'max':
                    max_bbox_size = (bbox[2] - bbox[0]) * ((bbox[3] - bbox[1]))
                elif self.select_type == 'center':
                    max_bbox_size = (abs((bbox[2] + bbox[0]) / 2 - shape_raw[1]/2)) * -1
                
            if max_idx != -1:
                person_count = 1

            if max_idx != -1:
                person = {}
                person['bbox'] = results[max_idx, :5]
                person['track_id'] = int(0)
                person_results.append(person)

            for i in range(results.shape[0]):
                bbox = results[i]
                if (bbox[-1] + 1 in cat_id) and (bbox[-2] > self.threshold_conf):
                    if self.select_type == 'max':
                        bbox_size = (bbox[2] - bbox[0]) * ((bbox[3] - bbox[1]))
                    elif self.select_type == 'center':
                        bbox_size = (abs((bbox[2] + bbox[0]) / 2 - shape_raw[1]/2)) * -1
                    if i != max_idx and bbox_size > max_bbox_size * self.threshold_multi_persons and bbox_size < max_bbox_size:
                        person_count += 1
                        if not single_person:
                            person = {}
                            person['bbox'] = results[i, :5]
                            person['track_id'] = int(person_count - 1)
                            person_results.append(person)                   
            return person_results
        else:
            return None
        

    def postprocess_threading(self, outputs, shape_raw, person_results, i, single_person=True, **kwargs):
        result = self.postprocess(outputs[i], shape_raw[i], cat_id=self.cat_id)
        result = self.process_results(result, shape_raw[i], cat_id=self.cat_id, single_person=single_person)
        if result is not None and len(result) != 0:
            person_results[i] = result


    def forward(self, img, shape_raw, **kwargs):
        """
        Performs inference using an ONNX model and returns the output image with drawn detections.

        Returns:
            output_img: The output image with drawn detections.
        """
        if isinstance(img, torch.Tensor):
            img = img.cpu().numpy()
            shape_raw = shape_raw.cpu().numpy()

        outputs = self.session.run(None, {self.session.get_inputs()[0].name: img})[0]
        person_results = [[{'bbox': np.array([0., 0., 1.*shape_raw[i][1], 1.*shape_raw[i][0], -1]), 'track_id': -1}] for i in range(len(outputs))]

        for i in range(len(outputs)):
            self.postprocess_threading(outputs, shape_raw, person_results, i, **kwargs)         
        return person_results


class ViTPose(SimpleOnnxInference):
    def __init__(self, checkpoint, device='cuda', **kwargs):
        super(ViTPose, self).__init__(checkpoint, device=device)

    def forward(self, img, center, scale, **kwargs):
        heatmaps = self.session.run([], {self.session.get_inputs()[0].name: img})[0]
        points, prob = keypoints_from_heatmaps(heatmaps=heatmaps,
                                            center=center,
                                            scale=scale*200,
                                            unbiased=True, 
                                            use_udp=False)
        return np.concatenate([points, prob], axis=2)


    @staticmethod
    def preprocess(img, bbox=None, input_resolution=(256, 192), rescale=1.25, mask=None, **kwargs):
        if bbox is None or bbox[-1] <= 0 or (bbox[2] - bbox[0]) < 10 or (bbox[3] - bbox[1]) < 10:
            bbox = np.array([0, 0, img.shape[1], img.shape[0]])
        
        bbox_xywh = bbox
        if mask is not None:
            img = np.where(mask>128, img, mask)

        if isinstance(input_resolution, int):
            center, scale = bbox_from_detector(bbox_xywh, (input_resolution, input_resolution), rescale=rescale)
            img, new_shape, old_xy, new_xy = crop(img, center, scale, (input_resolution, input_resolution))
        else:
            center, scale = bbox_from_detector(bbox_xywh, input_resolution, rescale=rescale)
            img, new_shape, old_xy, new_xy = crop(img, center, scale, (input_resolution[0], input_resolution[1]))

        IMG_NORM_MEAN = np.array([0.485, 0.456, 0.406])
        IMG_NORM_STD = np.array([0.229, 0.224, 0.225])
        img_norm = (img / 255. - IMG_NORM_MEAN) / IMG_NORM_STD
        img_norm = img_norm.transpose(2, 0, 1).astype(np.float32)
        return img_norm, np.array(center), np.array(scale)


class Pose2d:
    def __init__(self, checkpoint, detector_checkpoint=None, device='cuda', **kwargs):

        if detector_checkpoint is not None:
            self.detector = Yolo(detector_checkpoint, device)
        else:
            self.detector = None

        self.model = ViTPose(checkpoint, device)
        self.device = device

    def load_images(self, inputs):
        """
        Load images from various input types.
        
        Args:
            inputs (Union[str, np.ndarray, List[np.ndarray]]): Input can be file path, 
                     single image array, or list of image arrays
            
        Returns:
            List[np.ndarray]: List of RGB image arrays
            
        Raises:
            ValueError: If file format is unsupported or image cannot be read
        """
        if isinstance(inputs, str):
            if inputs.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                cap = cv2.VideoCapture(inputs)
                frames = []
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                cap.release()
                images = frames
            elif inputs.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                img = try_load_image(inputs)
                if img is None:
                    raise ValueError(f"Cannot read image: {inputs}")
                images = [img]
            else:
                raise ValueError(f"Unsupported file format: {inputs}")
                
        elif isinstance(inputs, np.ndarray):
            images = [cv2.cvtColor(image, cv2.COLOR_BGR2RGB) for image in inputs]
        elif isinstance(inputs, list):
            images = [cv2.cvtColor(image, cv2.COLOR_BGR2RGB) for image in inputs]
        return images

    def __call__(
        self, 
        inputs: Union[str, np.ndarray, List[np.ndarray]],
        return_image: bool = False,
        **kwargs
    ):
        """
        Process input and estimate 2D keypoints.
        
        Args:
            inputs (Union[str, np.ndarray, List[np.ndarray]]): Input can be file path,
                     single image array, or list of image arrays
            **kwargs: Additional arguments for processing
            
        Returns:
            np.ndarray: Array of detected 2D keypoints for all input images
        """
        images = self.load_images(inputs)
        H, W = images[0].shape[:2]
        if self.detector is not None:
            bboxes = []
            for _image in images:
                img, shape = self.detector.preprocess(_image)
                bboxes.append(self.detector(img[None], shape[None])[0][0]["bbox"])
        else:
            bboxes = [None] * len(images)

        kp2ds = []
        for _image, _bbox in zip(images, bboxes):
            img, center, scale = self.model.preprocess(_image, _bbox)
            kp2ds.append(self.model(img[None], center[None], scale[None]))
        kp2ds = np.concatenate(kp2ds, 0)
        metas = load_pose_metas_from_kp2ds_seq(kp2ds, width=W, height=H)
        return metas