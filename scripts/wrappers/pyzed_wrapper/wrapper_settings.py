
import pyzed.sl as sl

class InputParameters(object):
    ID = "id"
    SERIAL = "serial"
    SVO = "svo"
    STREAM = "stream"

    def __init__(self, type="id"):
        self.type = type

        if self.type == self.ID:
            self.id = 0
        elif self.type == self.SERIAL:
            self.serial_number = 0
        elif self.type == self.SVO:
            self.svo_input_filename = "./output.svo"
          

        elif self.type == self.STREAM:
            self.sender_ip = "127.0.0.1"
            self.port = 30000

        return


class ImageRetrievalParameters(object):

    def __init__(self):
        self.view = sl.VIEW.LEFT
        # self.view=sl.VIEW.SIDE_BY_SIDE
        self.type = sl.MEM.CPU
        self.resolution = sl.Resolution(0, 0)
        return


    def get(self):
        return {
            "view": self.view,
            "type": self.type,
            "resolution": self.resolution}


class MeasureRetrievalParameters(object):

    def __init__(self):
        self.measure = sl.MEASURE.DEPTH_U16_MM
        self.type = sl.MEM.CPU # MEM.GPU removes from stereolabs side
        self.resolution = sl.Resolution(0, 0)
        return


    def get(self):
        return {
            "measure": self.measure,
            "type": self.type,
            "resolution": self.resolution}


class CustomInitialParameters(object):
    def __init__(self):
        self.DEPTH_MODE = sl.DEPTH_MODE.NEURAL
        self.RESOLUTION = sl.RESOLUTION.HD720
        self.COORDINATE_UNITS = sl.UNIT.MILLIMETER
        self.CAMERA_FPS = 30
        self.DEPTH_MIN_DIST = 150
        self.DEPTH_MAX_DIST = 3000
        return
    

    def get(self):
        return {
            'depth_mode': self.DEPTH_MODE,
            'coordinate_units': self.COORDINATE_UNITS,
            'camera_fps': self.CAMERA_FPS,
            'depth_minimum_distance': self.DEPTH_MIN_DIST,
            'depth_maximum_distance': self.DEPTH_MAX_DIST,
            'camera_resolution': self.RESOLUTION
        }
    
class CustomRuntimeParameters(object):
    def __init__(self):
        self.FILL_DEPTH = True

        return
    

    def get(self):
        return {
            'enable_fill_mode': self.FILL_DEPTH,
        }