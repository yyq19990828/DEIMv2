"""
DEIMv2: Real-Time Object Detection Meets DINOv3
Copyright (c) 2025 The DEIMv2 Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from D-FINE (https://github.com/Peterande/D-FINE)
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
"""

import collections
import contextlib
import os
import time
from collections import OrderedDict

import cv2  # Added for video processing
import numpy as np
import tensorrt as trt
import torch
import torchvision.transforms as T
from PIL import Image, ImageDraw


class TimeProfiler(contextlib.ContextDecorator):
    def __init__(self):
        self.total = 0

    def __enter__(self):
        self.start = self.time()
        return self

    def __exit__(self, type, value, traceback):
        self.total += self.time() - self.start

    def reset(self):
        self.total = 0

    def time(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return time.time()

class TRTInference(object):
    def __init__(self, engine_path, device='cuda:0', backend='torch', max_batch_size=32, verbose=False):
        self.engine_path = engine_path
        self.device = device
        self.backend = backend
        self.max_batch_size = max_batch_size

        self.logger = trt.Logger(trt.Logger.VERBOSE) if verbose else trt.Logger(trt.Logger.INFO)

        self.engine = self.load_engine(engine_path)
        self.context = self.engine.create_execution_context()
        self.bindings = self.get_bindings(self.engine, self.context, self.max_batch_size, self.device)
        self.bindings_addr = OrderedDict((n, v.ptr) for n, v in self.bindings.items())
        self.input_names = self.get_input_names()
        self.output_names = self.get_output_names()
        self.time_profile = TimeProfiler()

    def load_engine(self, path):
        trt.init_libnvinfer_plugins(self.logger, '')
        with open(path, 'rb') as f, trt.Runtime(self.logger) as runtime:
            return runtime.deserialize_cuda_engine(f.read())

    def get_input_names(self):
        names = []
        for _, name in enumerate(self.engine):
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                names.append(name)
        return names

    def get_output_names(self):
        names = []
        for _, name in enumerate(self.engine):
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
                names.append(name)
        return names

    def get_bindings(self, engine, context, max_batch_size=32, device=None) -> OrderedDict:
        Binding = collections.namedtuple('Binding', ('name', 'dtype', 'shape', 'data', 'ptr'))
        bindings = OrderedDict()

        for i, name in enumerate(engine):
            shape = engine.get_tensor_shape(name)
            dtype = trt.nptype(engine.get_tensor_dtype(name))

            if shape[0] == -1:
                shape[0] = max_batch_size
                if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                    context.set_input_shape(name, shape)

            data = torch.from_numpy(np.empty(shape, dtype=dtype)).to(device)
            bindings[name] = Binding(name, dtype, shape, data, data.data_ptr())

        return bindings

    def run_torch(self, blob):
        for n in self.input_names:
            if self.bindings[n].shape != blob[n].shape:
                self.context.set_input_shape(n, blob[n].shape)
                self.bindings[n] = self.bindings[n]._replace(shape=blob[n].shape)

            assert self.bindings[n].data.dtype == blob[n].dtype, '{} dtype mismatch'.format(n)

        self.bindings_addr.update({n: blob[n].data_ptr() for n in self.input_names})
        self.context.execute_v2(list(self.bindings_addr.values()))
        outputs = {n: self.bindings[n].data for n in self.output_names}

        return outputs

    def __call__(self, blob):
        if self.backend == 'torch':
            return self.run_torch(blob)
        else:
            raise NotImplementedError("Only 'torch' backend is implemented.")

    def synchronize(self):
        if self.backend == 'torch' and torch.cuda.is_available():
            torch.cuda.synchronize()

def draw(images, labels, boxes, scores, thrh=0.4):
    for i, im in enumerate(images):
        draw = ImageDraw.Draw(im)
        scr = scores[i]
        lab = labels[i][scr > thrh]
        box = boxes[i][scr > thrh]
        scrs = scr[scr > thrh]

        for j, b in enumerate(box):
            draw.rectangle(list(b), outline='red')
            draw.text(
                (b[0], b[1]),
                text=f"{lab[j].item()} {round(scrs[j].item(), 2)}",
                fill='blue',
            )

    return images

def process_image(m, file_path, device, size=(640, 640), model_size='s'):
    im_pil = Image.open(file_path).convert('RGB')
    w, h = im_pil.size
    orig_size = torch.tensor([w, h])[None].to(device)

    transforms = T.Compose([
        T.Resize(size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
                if model_size not in ['atto', 'femto', 'pico', 'n'] 
                else T.Lambda(lambda x: x)
    ])
    im_data = transforms(im_pil)[None]

    blob = {
        'images': im_data.to(device),
        'orig_target_sizes': orig_size.to(device),
    }

    output = m(blob)
    result_images = draw([im_pil], output['labels'], output['boxes'], output['scores'])
    result_images[0].save('trt_result.jpg')
    print("Image processing complete. Result saved as 'result.jpg'.")

def process_video(m, file_path, device, size=(640, 640), model_size='s'):
    cap = cv2.VideoCapture(file_path)

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('trt_result.mp4', fourcc, fps, (orig_w, orig_h))

    transforms = T.Compose([
        T.Resize(size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
                    if model_size not in ['atto', 'femto', 'pico', 'n'] 
                    else T.Lambda(lambda x: x)
    ])

    frame_count = 0
    print("Processing video frames...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to PIL image
        frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        w, h = frame_pil.size
        orig_size = torch.tensor([w, h])[None].to(device)

        im_data = transforms(frame_pil)[None]

        blob = {
            'images': im_data.to(device),
            'orig_target_sizes': orig_size.to(device),
        }

        output = m(blob)

        # Draw detections on the frame
        result_images = draw([frame_pil], output['labels'], output['boxes'], output['scores'])

        # Convert back to OpenCV image
        frame = cv2.cvtColor(np.array(result_images[0]), cv2.COLOR_RGB2BGR)

        # Write the frame
        out.write(frame)
        frame_count += 1

        if frame_count % 10 == 0:
            print(f"Processed {frame_count} frames...")

    cap.release()
    out.release()
    print("Video processing complete. Result saved as 'result_video.mp4'.")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-trt', '--trt', type=str, required=True)
    parser.add_argument('-i', '--input', type=str, required=True)
    parser.add_argument('-d', '--device', type=str, default='cuda:0')
    parser.add_argument('-s', '--size', type=int, required=True, help='input size, e.g., 640')
    parser.add_argument('-ms', '--model-size', type=str, required=True, choices=['atto', 'femto', 'pico', 'n', 's', 'm', 'l', 'x'])


    args = parser.parse_args()

    m = TRTInference(args.trt, device=args.device)
    size = (args.size,) * 2

    file_path = args.input
    if os.path.splitext(file_path)[-1].lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
        # Process as image
        process_image(m, file_path, args.device, size, args.model_size)
    else:
        # Process as video
        process_video(m, file_path, args.device, size, args.model_size)
