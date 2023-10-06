import cv2
import numpy as np
import onnxruntime as rt
import os

class Mask_det(object):

    def __init__(self, onnx_file=None):
        if onnx_file:
            self.model_path = onnx_file
        else:
            self.model_path = './models/yolov5s.onnx'

        self.sess = rt.InferenceSession(self.model_path)
        self.size_h = self.sess.get_inputs()[0].shape[2]
        self.size_w = self.sess.get_inputs()[0].shape[3]
        self.input_name = self.sess.get_inputs()[0].name


    def nms(self, dets, scores, iou_threshold):
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]

        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]
        ndets = dets.shape[0]

        suppressed = np.zeros(ndets, dtype=np.int64)
        keep = np.zeros(ndets, dtype=np.int64)
        
        num_to_keep = 0
        for _i in range(ndets):
            i = order[_i]
            if suppressed[i] == 1:
                continue
            keep[num_to_keep] = i
            num_to_keep += 1
            ix1 = x1[i]
            iy1 = y1[i]
            ix2 = x2[i]
            iy2 = y2[i]
            iarea = areas[i]
            for _j in range(_i+1, ndets):
                j = order[_j]
                if suppressed[j] == 1:
                    continue
                xx1 = max(ix1, x1[j])
                yy1 = max(iy1, y1[j])
                xx2 = min(ix2, x2[j])
                yy2 = min(iy2, y2[j])

                w = max(0, xx2 - xx1)
                h = max(0, yy2 - yy1)
                inter = w * h;
                ovr = inter / (iarea + areas[j] - inter);
                if (ovr > iou_threshold):
                    suppressed[j] = 1

        return keep[:num_to_keep]


    def non_max_suppression(self, prediction, conf_thres=0.7, iou_thres=0.6):

        nc = prediction.shape[2] - 5  
        xc = prediction[..., 4] > conf_thres  

        min_wh, max_wh = 2, 4096  
        max_det = 300  
        max_nms = 30000  

        output = [np.zeros((0, 6))] * prediction.shape[0]
        for xi, x in enumerate(prediction):  # image index, image inference
            x = x[xc[xi]]  # confidence

            if not x.shape[0]:
                continue

            # Compute conf
            x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

            # Box (center x, center y, width, height) to (x1, y1, x2, y2)
            box = self.xywh2xyxy(x[:, :4])

            conf = np.max(x[:, 5:], axis=1, keepdims=True)
            j = np.argmax(x[:, 5:], axis=1)
            j = j.reshape(j.shape[0], -1)
            x = np.concatenate([box, conf, j], axis=1)
            x = x[conf.reshape(-1, ) > conf_thres]

            # Check shape
            n = x.shape[0]  # number of boxes
            if not n:  # no boxes
                continue
            elif n > max_nms:  # excess boxes
                x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence

            # Batched NMS
            c = x[:, 5:6] * (max_wh)  # classes
            boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
            i = self.nms(boxes, scores, iou_thres)
            if i.shape[0] > max_det:  # limit detections
                i = i[:max_det]

            output[xi] = x[i]

        return output


    def letterbox(self, img, new_shape=(640, 640), color=(114, 114, 114)):
        # Resize and pad image while meeting stride-multiple constraints
        shape = img.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border

        return img


    def clip_coords(self, boxes, img_shape):
        boxes[:, 0].clip(0, img_shape[1])  # x1
        boxes[:, 1].clip(0, img_shape[0])  # y1
        boxes[:, 2].clip(0, img_shape[1])  # x2
        boxes[:, 3].clip(0, img_shape[0])  # y2


    def scale_coords(self, img1_shape, coords, img0_shape, ratio_pad=None):
        # Rescale coords (xyxy) from img1_shape to img0_shape
        if ratio_pad is None:  # calculate from img0_shape
            gain = max(img1_shape) / max(img0_shape)  # gain  = old / new
            pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
        else:
            gain = ratio_pad[0][0]
            pad = ratio_pad[1]

        coords[:, [0, 2]] -= pad[0]  # x padding
        coords[:, [1, 3]] -= pad[1]  # y padding
        coords[:, :4] /= gain
        self.clip_coords(coords, img0_shape)
        return coords


    def xywh2xyxy(self, x):
        # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
        y = np.zeros_like(x)
        y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
        y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
        y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
        y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
        return y


    def numpy2dic(self, boxes):

        res = []
        for box in boxes:
            x1, y1, x2, y2, conf, class_name = box
            x1, y1, x2, y2, class_name = int(x1), int(y1), int(x2), int(y2), int(class_name)

            if class_name == 1:
                class_name = 0
            elif class_name == 0:
                class_name = 1

            dic = {}
            dic['location'] = {}
            dic['location']['left'] = x1
            dic['location']['top'] = y1
            dic['location']['width'] = x2 - x1
            dic['location']['height'] = y2 - y1

            dic['face_mask'] = {}
            dic['face_mask']['score'] = conf
            dic['face_mask']['name'] = class_name

            res.append(dic)

        return res


    def infer(self, image, conf_thres=0.7, iou_thres=0.6):

        h, w, _ = image.shape
        image_resized = self.letterbox(image, (self.size_w, self.size_h))
        image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)

        image_in = np.transpose(image_rgb, (2, 0, 1)).astype(np.float32)  
        image_in = np.expand_dims(image_in, axis=0)
        image_in /= 255.0

        outputs = self.sess.run(None, {self.input_name: image_in})

        detections = outputs[0]
        detections = self.non_max_suppression(detections, conf_thres=conf_thres, iou_thres=iou_thres)

        scaled_coords = self.scale_coords((self.size_w, self.size_h), detections[0], (h, w))

        return self.numpy2dic(scaled_coords)


def display(detections, image):

    # class_names = ['mask', 'no_mask', 'uncertain']
    class_names = ['no_mask', 'mask', 'uncertain']

    for box in detections:

        x1 = box['location']['left']
        y1 = box['location']['top']
        x2 = box['location']['left'] + box['location']['width']
        y2 = box['location']['top'] + box['location']['height']

        class_name = box['face_mask']['name']
        conf = box['face_mask']['score']

        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)

        text = '%s %.2f' % (class_names[class_name], conf)
        cv2.putText(image, text, (x1 + 3, y1 - 4), 3, 1, (0, 255, 255))


    return image


if __name__ == "__main__":

    import time
    import json

    ct = 0.7
    it = 0.6

    model = Mask_det()
    video_path = "../input.mp4"
    image_dir = "../images"
    cmd = "ffmpeg -i %s -vf fps=1 %s/out%%d.jpg" % (video_path,image_dir)
    os.system(cmd)
    for filename in os.listdir(image_dir):
        filepath = os.path.join(image_dir, filename)
        if os.path.isfile(filepath):
            print(f'File: {filepath}')
            image = cv2.imread(filepath)
            tic = time.time()
            detections = model.infer(image, conf_thres=ct, iou_thres=it)
            elapsed = time.time() - tic
            print(elapsed)
            print(detections)

            img_vis = display(detections, image)
            save_path = "./mask_%s.jpg" % filename
            cv2.imwrite(save_path, img_vis)

            with open('res.json', 'w', encoding='utf-8') as f:
                json.dump(detections, f, ensure_ascii=False, indent=4)
    # image_path = './images/img.jpg'
    # # image = cv2.imread(image_path)

    # tic = time.time()
    # detections = model.infer(image, conf_thres=ct, iou_thres=it)
    # elapsed = time.time() - tic
    # print(elapsed)
    # print(detections)

    # img_vis = display(detections, image)
    # cv2.imwrite('./mask.jpg', img_vis)

    # with open('res.json', 'w', encoding='utf-8') as f:
    #     json.dump(detections, f, ensure_ascii=False, indent=4)


    # image_list = glob.glob('/media/chen/38EC01F3EC01AC66/mask/test/*.jpg')
    # print(len(image_list))

    # times = []
    # for image_path in tqdm.tqdm(image_list):
    #     tic = time.time()
    #     image = cv2.imread(image_path)
    #     detections = model.infer(image)
    #     elapsed = time.time() - tic
    #     times.append(elapsed)

    #     img_vis = display(detections, image)

    #     cv2.imwrite(os.path.join('vis', os.path.basename(image_path)), img_vis)

    # print(np.mean(times[1:]))


