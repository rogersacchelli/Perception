import argparse
import cv2
from line_detector import *
from LaneClass import Line, ImageLine


# Variables definitions

DEFAULT_IMAGE_SHAPE = (720, 1280, 3)


def parse():
    parser = argparse.ArgumentParser(description='Perception 0.0.1')
    input = parser.add_mutually_exclusive_group(required=True)
    features = parser.add_argument_group()

    input.add_argument("-v", "--video_input", type=str, help='Video Input File')
    input.add_argument("-d", "--dir_image", type=str, help='Directory of Images')
    input.add_argument("-cam", "--camera", type=str, help='Camera Input')

    features.add_argument("-a", "--all_features", help="Process Detection of all features", action="store_true")
    features.add_argument("-l", "--lane", help="Add Lane Detection", action="store_true")
    features.add_argument("-c", "--cars", help="Add Lane Detection", action="store_true")

    args = parser.parse_args()
    return args


def open_stream(input_type):
    """

    :param input_type:
    :return: Returns an iterator for image processing
    """

    if input_type.video_input:
        return cv2.VideoCapture(input_type.video_input)
    elif input_type.dir_image:
        raise NotImplemented
    elif input_type.camera:
        return cv2.VideoCapture(0)


def main():
    arguments = parse()
    input_iterator = open_stream(arguments)

    if arguments.lane:
        ret, mtx, dist, rvecs, tvecs = load_camera()
        line_image = ImageLine(np.zeros(shape=DEFAULT_IMAGE_SHAPE, dtype=np.float32), ret, mtx, dist, rvecs, tvecs)
        line_info = Line()

    while input_iterator.isOpened():
        ret,  frame = input_iterator.read()
        if ret:
            if arguments.lane:
                # Do lane lines processing
                line_image.image = frame
                line_image, line_info = line_detector(line_image, line_info)
            elif arguments.cars:
                # Car Detection
                # TODO: implement car detection
                raise NotImplemented
            cv2.imshow('final', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        else:
            input_iterator.release()
            cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
