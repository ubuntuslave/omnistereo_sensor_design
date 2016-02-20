import cv2
from omnistereo import common_cv

class WebcamLive:
    def __init__(self, cam_index=0, mirror_image=False, win_name="Webcam Video", file_name="", **kwargs):
        self.cam_index = cam_index
        self.mirror_image = mirror_image
        if file_name == "":
            import time
            localtime = time.localtime()
            time_str = time.strftime("%Y-%m-%d-%H-%M-%S", localtime)
            file_name = "img-" + time_str  # Use current date and time
        self.file_name = file_name
        self.cam_model = kwargs.get("cam_model", "GENERIC")
        self.save_key = kwargs.get("save_key", 's')
        self.show_img = kwargs.get("show_img", True)
        # Start Frame Capturing
        self.capture = cv2.VideoCapture(self.cam_index)
        # Setting desired frame width and height (if any particular known model)
        if self.cam_model == "CHAMELEON":
            self.capture.set(cv2.CAP_PROP_MODE, cv2.CAP_MODE_GRAY)
#             self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1296)  # Is it 1280 or 1296?
            self.needs_bayer_conversion = True
        elif self.cam_model == "BLACKFLY":
            # self.capture.set(cv2.CAP_PROP_MODE, cv2.CAP_MODE_GRAY)
            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            self.capture.set(cv2.CAP_PROP_FPS, -1)
            self.needs_bayer_conversion = True
        else:
            self.needs_bayer_conversion = False

        self.print_cam_info()
        if self.capture.isOpened():
            # For some reason, USB3 Point Grey cameras require a few trials to read the capture
            success = False
            failure_count = 0
            while not success:
                success, frame = self.capture.read()
                cv2.waitKey(1000)
                if not success:
                    failure_count += 1
                    print("Failed to read (%d). Try reading again" % (failure_count))
            if success:
                h, w = frame.shape[0:2]
                self.channels = frame.ndim
                self.img_size = (w, h)
                print("Successfully initialize video source %d using %d color channels" % (self.cam_index, self.channels))
        else:
            print("Failed to initialize video source %d" % (self.cam_index))

        self.win_name = win_name
        if self.show_img:
            cv2.namedWindow(self.win_name, cv2.WINDOW_NORMAL)
            # cv2.namedWindow(win_name, cv2.WINDOW_AUTOSIZE)
            # cv2.namedWindow(win_name, cv2.WINDOW_OPENGL)
            self.win_handler = common_cv.PointClicker(self.win_name)

    def print_cam_info(self):
#         capture_mode = self.capture.get(cv2.CAP_PROP_MODE)
#         print("Capture mode:", capture_mode)
#         codec = self.capture.get(cv2.CAP_PROP_FOURCC)
#         print("4-character code of codec:", codec)
        fps = self.capture.get(cv2.CAP_PROP_FPS)
        print("Framerate = %0.2f FPS" % (fps))
        gain = self.capture.get(cv2.CAP_PROP_GAIN)
        print("Gain = %0.2f" % (gain))

    def demo_live(self):
        success, _ = self.get_single_frame()
        while success and self.capture.isOpened():
            success, frame = self.get_single_frame()
#             self.capture.get(cv2.CAP_PVAPI_PIXELFORMAT_BAYER8)
#             self.capture.get(cv2.CAP_PVAPI_PIXELFORMAT_BAYER16)
#             self.capture.get(cv2.CAP_PVAPI_PIXELFORMAT_BGR24)
#             self.capture.get(cv2.CAP_PROP_MONOCHROME)

#===============================================================================
# # The PointGrey BlackFly that reads as three 8-bit channels in Python's Numpy array of shape (rows, cols, 3).
# # However, each cell (pixel) has the same value, so I end up with a gray-scale image-like by default.
# # I tried several conversions of color, but Bayer cannot go from 3 channels to anything else.
# # Does anyone know which property to set for this VideoCapture object or any other suggestion? I'm not sure which Video Data Output it's using, perhaps it's 24-bit digital data it's giving 24-bits per pixel as (8,8,8), but maybe not. How to find out the video format within OpenCV? If I ask for get(CAP_PVAPI_PIXELFORMAT_RGB24) it give 15.0, get(CAP_PVAPI_PIXELFORMAT_BAYER16) gives 480.0, and get(CAP_PVAPI_PIXELFORMAT_BAYER8) gives 640.0. Any other PIXELFORMAT gives -1. I don't understand it.
#===============================================================================


    def get_single_frame(self, show_img=True):
        frame = None
        success = False
        if (not self.show_img) and show_img:
            # Create window
            cv2.namedWindow(self.win_name, cv2.WINDOW_NORMAL)
            self.win_handler = common_cv.PointClicker(self.win_name)
        elif self.show_img and not show_img:
            # Destroy window
            cv2.destroyWindow(self.win_name)

        self.show_img = show_img
        try:
            # Grabs, decodes and returns the next video frame
            success, frame_raw = self.capture.read()
            if success:
                if self.needs_bayer_conversion:
                    if self.cam_model == "CHAMELEON":
                        frame = cv2.cvtColor(frame_raw, cv2.COLOR_BAYER_GR2BGR)  # TODO: figure out the right scheme for the 2nd COLOR conversion chameleon
                    elif self.cam_model == "BLACKFLY":
                        frame = cv2.cvtColor(frame_raw[..., 0], cv2.COLOR_BAYER_BG2BGR)
                else:
                    frame = frame_raw

                if self.mirror_image:
                    frame = frame[:, -1::-1]

                if self.show_img:
                    cv2.imshow(self.win_name, frame)

                    ch_pressed_waitkey = cv2.waitKey(1)

                    if (ch_pressed_waitkey & 255) == ord(self.save_key):
                        self.win_handler.save_image(frame, img_name=self.file_name)
                    if ch_pressed_waitkey == 27:  # Pressing the Escape key breaks the loop
                        success = False
        except:
            print("Failed to read frame!")

        return success, frame

import numpy as np

class WebcamLiveDuo:

    def __init__(self, cam_indices, mirror_image=False, win_name="Webcam Video", file_name="", cam_models=[], save_keys=[]):
        self.cams = []
        for i, c_model, s_key in zip(cam_indices, cam_models, save_keys):
            self.cams.append(WebcamLive(cam_index=i, mirror_image=mirror_image, win_name="%s-%d" % (win_name, i), file_name="cam_%d-%s" % (i, file_name), cam_model=c_model, save_key=s_key))

        self.success_list = np.zeros(len(cam_indices), dtype="bool")
        self.open_list = np.zeros(len(cam_indices), dtype="bool")

    def demo_live(self):
        for i, c in enumerate(self.cams):
            self.success_list[i], frame = c.get_single_frame()
            self.open_list[i] = c.capture.isOpened()

        while np.all(self.success_list) and np.all(self.open_list):
            for i, c in enumerate(self.cams):
                self.success_list[i], frame = c.get_single_frame()
                self.open_list[i] = c.capture.isOpened()

if __name__ == '__main__':
#     cam = WebcamLive(cam_index=0, mirror_image=False, file_name="", cam_model="CHAMELEON")
    cam = WebcamLive(cam_index=0, mirror_image=False, file_name="", cam_model="BLACKFLY")
#     cam = WebcamLive(cam_index=0, mirror_image=False, file_name="chessboard")
#     cam = WebcamLiveDuo(cam_indices=[0, 1], mirror_image=False, file_name="duo", cam_models=["CHAMELEON", "CHAMELEON"], save_keys=['a', 's'])

    cam.demo_live()
    cv2.destroyAllWindows()
