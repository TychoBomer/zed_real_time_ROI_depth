"""
Based on Stereolabs zed API
https://github.com/stereolabs/zed-examples/
https://www.stereolabs.com/docs/api/python/

Image and video export is not included, to avoid dependencies on e.g. OpenCV.
You can write this yourself from the numpy arrays.

TODO: incorporate proper logging on all errors and prints
"""

import pyzed.sl as sl
from . import wrapper_settings as ws


class Wrapper(object):
    """
    Wrapper for ZED API (v3.6) to collect basic functionality in a single class.
    """


    def __init__(self, input_type):
        """
        Configures only default parameters.
        Changes to defaults parameters must be set manually.

        =INPUT=
            input_type - string
                Either "id", "serial", "svo", or "stream"
        """

        self.input_parameters = ws.InputParameters(input_type)
        self.initial_parameters = sl.InitParameters()
        self.streaming_parameters = sl.StreamingParameters()
        self.recording_parameters = sl.RecordingParameters()
        self.runtime_parameters = sl.RuntimeParameters()
        self.image_retrieval_parameters = ws.ImageRetrievalParameters()
        self.measure_retrieval_parameters = ws.MeasureRetrievalParameters()
        self.custom_initial_parameters = ws.CustomInitialParameters()
        self.custom_runtime_parameters = ws.CustomRuntimeParameters()
        self.camera = sl.Camera()
        self.mat_image = sl.Mat()
        self.mat_measure = sl.Mat()
        self.grab_status = None
        self.timestamp_nanoseconds = 0

        

        # Numpy output arrays
        self.output_image = None
        self.output_measure = None

        # Override initial parameters with custom parameters
        for param, value in self.custom_initial_parameters.get().items():
            setattr(self.initial_parameters, param, value)

        # Override initial runtime params with given custom paramers
        for param, value in self.custom_runtime_parameters.get().items():
            setattr(self.runtime_parameters, param, value)

        

    def _set_input_type(self):
        """
        Depending on the input type set at instance creation
        """

        # Update InputType
        if self.input_parameters.type == self.input_parameters.ID:
            self.initial_parameters.set_from_camera_id(
                self.input_parameters.id)
        elif self.input_parameters.type == self.input_parameters.SERIAL:
            self.initial_parameters.set_from_serial_number(
                self.input_parameters.serial_number)
        elif self.input_parameters.type ==  self.input_parameters.SVO:
            self.initial_parameters.set_from_svo_file(
                self.input_parameters.svo_input_filename)
        elif self.input_parameters.type == self.input_parameters.STREAM:
            self.initial_parameters.set_from_stream(
                self.input_parameters.sender_ip,
                self.input_parameters.port)
        return

    
    def get_intrinsic(self):
        """
        https://www.stereolabs.com/docs/api/python/classpyzed_1_1sl_1_1CameraParameters.html

        =OUTPUT=
            intrinsic_left, intrinsic_right - CameraParameters (see link)
        """

        camera_information = self.camera.get_camera_information()
        intrinsic_left = camera_information.camera_configuration.calibration_parameters_raw.left_cam
        intrinsic_right = camera_information.camera_configuration.calibration_parameters_raw.right_cam

        return intrinsic_left, intrinsic_right


    def open_input_source(self):
        """
        Open the camera, stream, or file connection, depending on
        set_input_type.
        """
        self._set_input_type()

        status = self.camera.open(self.initial_parameters)
        if status != sl.ERROR_CODE.SUCCESS: 
            raise Exception("Failed to open input source '{0}': {1}".format(
                self.input_parameters.type, status))
        return


    def close_input_source(self):
        """
        Free mat memory (CPU + GPU) and close camera connection.
        """
        # self.mat_image.free(sl.MEM.GPU)
        self.output_image = None
        self.mat_image.free(sl.MEM.CPU)
        # self.mat_measure.free(sl.MEM.GPU)
        self.output_measure = None
        self.mat_measure.free(sl.MEM.CPU)
        
        self.camera.close()
        return


    def start_stream(self):
        """
        Start streaming data. Must be called after open_input_source.
        """
        if not self.camera.is_opened():
            return

        status = self.camera.enable_streaming(self.streaming_parameters)
        if status != sl.ERROR_CODE.SUCCESS:
            raise Exception('Failed to start stream: {0}'.format(status))
        return


    def stop_stream(self):
        self.camera.disable_streaming()
        return


    def start_recording(self, filename="output.svo"):
        """
        Start recording data into an SVO file. Must be called after
        open_input_source. Frames are written using retrieve (on grab).
        Advised to set is_image and is_measure to False. The user is 
        responsible for grabbing frames at a sufficient pace.
        """
        if not self.camera.is_opened():
            return
        self.recording_parameters.video_filename = filename
        status = self.camera.enable_recording(self.recording_parameters)
        if status != sl.ERROR_CODE.SUCCESS:
            raise Exception('Failed to start recording: {0}'.format(status))
        return


    def stop_recording(self):
        self.camera.disable_recording()
        return


    def retrieve(self, is_image=True, is_measure=False):
        """
        Receive single captured image frame or image measure

        =NOTES=
            It is likely more efficient to obtain SIDE_BY_SIDE and then split
            the array, than calling retrieve image twice for left and right
            separately. However, obtaining only IMAGE_LEFT is likely more
            efficient than getting SIDE_BY_SIDE.

        TODO: there's various optimization that can be done here, such as using
        GPU and preventing unnecessary copies between CPU and GPU on calling
        retrieve_image.
        """

        self.grab_status = self.camera.grab(self.runtime_parameters)
        if self.grab_status == sl.ERROR_CODE.SUCCESS:
            if is_image:
                self.camera.retrieve_image(
                    self.mat_image,
                    **self.image_retrieval_parameters.get())

                # Put data in numpy array
                self.output_image = self.mat_image.get_data()
                self.timestamp_nanoseconds = self.mat_image.timestamp.data_ns

            if is_measure:
                self.camera.retrieve_measure(
                    self.mat_measure,
                    **self.measure_retrieval_parameters.get())

                # Put data in numpy array
                self.output_measure = self.mat_measure.get_data()

        else:
            print('Failed to grab camera frame.')
            return False

        return True


    def export_png(self, filename, is_image=True, is_measure=False):
        """
        Export last retrieved mat as png image.

        =INPUT=
            filename - string
                Path name + file name + .png extension
        """

        if self.grab_status != sl.ERROR_CODE.SUCCESS:
            return False

        if is_image:
            self.mat_image.write(filename)

        if is_measure:
            self.mat_measure.write(filename)

        return True


    def depth_at_xy(self, xpos:int = 0, ypos:int = 0):
        depth = self.output_measure[xpos,ypos]
        return depth
    
    