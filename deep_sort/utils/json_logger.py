import json
from os import makedirs
from os.path import exists, join
from datetime import datetime


class JsonMeta(object):
    HOURS = 3
    MINUTES = 59
    SECONDS = 59
    PATH_TO_SAVE = 'LOGS'
    DEFAULT_FILE_NAME = 'remaining'


class BaseJsonLogger(object):

    def dic(self):
        # returns dicts of objects
        out = {}
        for k, v in self.__dict__.items():
            if hasattr(v, 'dic'):
                out[k] = v.dic()
            elif isinstance(v, list):
                out[k] = self.list(v)
            else:
                out[k] = v
        return out

    def list(values):
        # applies the dic method on items in the list
        return [v.dic() if hasattr(v, 'dic') else v for v in values]


class Label(BaseJsonLogger):

    def __init__(self, category: str, confidence: float):
        self.category = category
        self.confidence = confidence


class Bbox(BaseJsonLogger):

    def __init__(self, bbox_id, top, left, width, height):
        self.labels = []
        self.bbox_id = bbox_id
        self.top = top
        self.left = left
        self.width = width
        self.height = height

    def add_label(self, category, confidence):
        # adds category and confidence only if top_k is not exceeded.
        self.labels.append(Label(category, confidence))

    def labels_full(self, value):
        return len(self.labels) == value


class Frame(BaseJsonLogger):

    def __init__(self, frame_id: int, timestamp: float = None):
        self.frame_id = frame_id
        self.timestamp = timestamp
        self.bboxes = []

    def add_bbox(self, bbox_id: int, top: int, left: int, width: int, height: int):
        bboxes_ids = [bbox.bbox_id for bbox in self.bboxes]
        if bbox_id not in bboxes_ids:
            self.bboxes.append(Bbox(bbox_id, top, left, width, height))
        else:
            raise ValueError("Frame with id: {} already has a Bbox with id: {}".format(self.frame_id, bbox_id))

    def add_label_to_bbox(self, bbox_id: int, category: str, confidence: float):
        bboxes = {bbox.id: bbox for bbox in self.bboxes}
        if bbox_id in bboxes.keys():
            res = bboxes.get(bbox_id)
            res.add_label(category, confidence)
        else:
            raise ValueError('the bbox with id: {} does not exists!'.format(bbox_id))


class BboxToJsonLogger(BaseJsonLogger):

    def __init__(self, top_k_labels: int = 1):
        self.frames = {}
        self.video_details = self.video_details = dict(frame_width=None, frame_height=None, frame_rate=None,
                                                       video_name=None)
        self.top_k_labels = top_k_labels
        self.start_time = datetime.now()

    def set_top_k(self, value):
        self.top_k_labels = value

    def frame_exists(self, frame_id: int) -> bool:
        return frame_id in self.frames.keys()

    def add_frame(self, frame_id: int, timestamp: float = None) -> None:
        if not self.frame_exists(frame_id):
            self.frames[frame_id] = Frame(frame_id, timestamp)
        else:
            raise ValueError("Frame id: {} already exists".format(frame_id))

    def bbox_exists(self, frame_id: int, bbox_id: int) -> bool:
        bboxes = []
        if self.frame_exists(frame_id=frame_id):
            bboxes = [bbox.bbox_id for bbox in self.frames[frame_id].bboxes]
        return bbox_id in bboxes

    def find_bbox(self, frame_id: int, bbox_id: int):
        if not self.bbox_exists(frame_id, bbox_id):
            raise ValueError("frame with id: {} does not contain bbox with id: {}".format(frame_id, bbox_id))
        bboxes = {bbox.bbox_id: bbox for bbox in self.frames[frame_id].bboxes}
        return bboxes.get(bbox_id)

    def add_bbox_to_frame(self, frame_id: int, bbox_id: int, top: int, left: int, width: int, height: int) -> None:
        if self.frame_exists(frame_id):
            frame = self.frames[frame_id]
            if not self.bbox_exists(frame_id, bbox_id):
                frame.add_bbox(bbox_id, top, left, width, height)
            else:
                raise ValueError(
                    "frame with frame_id: {} already contains the bbox with id: {} ".format(frame_id, bbox_id))
        else:
            raise ValueError("frame with frame_id: {} does not exist".format(frame_id))

    def add_label_to_bbox(self, frame_id: int, bbox_id: int, category: str, confidence: float):
        bbox = self.find_bbox(frame_id, bbox_id)
        if not bbox.labels_full(self.top_k_labels):
            bbox.add_label(category, confidence)
        else:
            raise ValueError("labels in frame_id: {}, bbox_id: {} is fulled".format(frame_id, bbox_id))

    def add_video_details(self, frame_width: int = None, frame_height: int = None, frame_rate: int = None,
                          video_name: str = None):
        self.video_details['frame_width'] = frame_width
        self.video_details['frame_height'] = frame_height
        self.video_details['frame_rate'] = frame_rate
        self.video_details['video_name'] = video_name

    def output(self):
        output = {'video_details': self.video_details}
        result = list(self.frames.values())
        output['frames'] = [item.dic() for item in result]
        return output

    def json_output(self, output_name):
        if not output_name.endswith('.json'):
            output_name += '.json'
        with open(output_name, 'w') as file:
            json.dump(self.output(), file)
        file.close()

    def set_start(self):
        self.start_time = datetime.now()

    def schedule_output_by_time(self, output_dir=JsonMeta.PATH_TO_SAVE, hours: int = 0, minutes: int = 0,
                                seconds: int = 60) -> None:
        end = datetime.now()
        interval = 0
        interval += abs(min([hours, JsonMeta.HOURS]) * 3600)
        interval += abs(min([minutes, JsonMeta.MINUTES]) * 60)
        interval += abs(min([seconds, JsonMeta.SECONDS]))
        diff = (end - self.start_time).seconds

        if diff > interval:
            output_name = self.start_time.strftime('%Y-%m-%d %H-%M-%S') + '.json'
            if not exists(output_dir):
                makedirs(output_dir)
            output = join(output_dir, output_name)
            self.json_output(output_name=output)
            self.frames = {}
            self.start_time = datetime.now()

    def schedule_output_by_frames(self, frames_quota, frame_counter, output_dir=JsonMeta.PATH_TO_SAVE):
        pass

    def flush(self, output_dir):
        filename = self.start_time.strftime('%Y-%m-%d %H-%M-%S') + '-remaining.json'
        output = join(output_dir, filename)
        self.json_output(output_name=output)
