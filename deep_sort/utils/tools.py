from functools import wraps
from time import time

#路径/格式判断
def is_video(ext: str):

    allowed_exts = ('.mp4', '.webm', '.ogg', '.avi', '.wmv', '.mkv', '.3gp')
    return any((ext.endswith(x) for x in allowed_exts))


def tik_tok(func):
    @wraps(func)
    def _time_it(*args, **kwargs):
        start = time()
        try:
            return func(*args, **kwargs)
        finally:
            end_ = time()
            print("time: {:.03f}s, fps: {:.03f}".format(end_ - start, 1 / (end_ - start)))

    return _time_it
