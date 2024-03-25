import argparse
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from shapely.geometry import Polygon
from shapely.geometry.point import Point

from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors

track_history = defaultdict(list)

current_region = None
counting_regions = [
    {
        "name": "Circle Region",
        "coordinates": Point(300, 300).buffer(200),    # Center and radious
        "counts": 0,
        "dragging": False,
        "region_color": (255, 42, 4),  # BGR Value
        "text_color": (255, 255, 255)  # Region Text Color
    },
    {
        "name": "Rectangle Region",
        "coordinates": Polygon([(200, 250), (440, 250), (440, 550), (200, 550)]),  # Polygon points
        "counts": 0,
        "dragging": False,
        "region_color": (37, 255, 225),
        "text_color": (0, 0, 0)
    }
]


def mouse_callback(event, x, y, flags, param):
    """
    Handles mouse events for region manipulation.

    Parameters:
        event (int): The mouse event type (e.g., cv2.EVENT_LBUTTONDOWN).
        x (int): The x-coordinate of the mouse pointer.
        y (int): The y-coordinate of the mouse pointer.
        flags (int): Additional flags passed by OpenCV.
        param: Additional parameters passed to the callback (not used in this function).

    Global Variables:
        current_region (dict): A dictionary representing the current selected region.

    Mouse Events:
        - LBUTTONDOWN: Initiates dragging for the region containing the clicked point.
        - MOUSEMOVE: Moves the selected region if dragging is active.
        - LBUTTONUP: Ends dragging for the selected region.

    Example:
        >>> cv2.setMouseCallback(window_name, mouse_callback)
    """
    global current_region

    # Mouse left button down event
    if event == cv2.EVENT_LBUTTONDOWN:
        for region in counting_regions:
            if region["coordinates"].contains(Point((x, y))):
                current_region = region
                current_region["dragging"] = True
                current_region["offset_x"] = x
                current_region["offset_y"] = y

    # Mouse move event
    elif event == cv2.EVENT_MOUSEMOVE:
        if current_region is not None and current_region["dragging"]:
            dx = x - current_region["offset_x"]
            dy = y - current_region["offset_y"]
            current_region["coordinates"] = Polygon(
                [(p[0] + dx, p[1] + dy) for p in current_region["coordinates"].exterior.coords]
            )
            current_region["offset_x"] = x
            current_region["offset_y"] = y

    # Mouse left button up event
    elif event == cv2.EVENT_LBUTTONUP:
        if current_region is not None and current_region["dragging"]:
            current_region["dragging"] = False


def run(weights="yolov8n.pt", source=None, device="cpu", view_result=False, save_result=False, classes=None):
    """
    Run Region counting on a video using YOLOv8 and ByteTrack.

    Supports movable region for real time counting inside specific area.
    Supports multiple regions counting.
    Regions can be polygon or circle.

    Args:
        weights (str): Model weights path.
        source (str): Video file path.
        device (str): processing device cpu, 0, 1
        view_result (bool): Show results.
        save_result (bool): Save results.
        classes (list): classes to detect and track
    """
    vid_frame_count = 0

    # Check source path
    if not Path(source).exists():
        raise FileNotFoundError(f"Source path '{source}' does not exist.")

    # Setup Model
    model = YOLO(weights)
    model.to("cuda") if device == "0" else model.to("cpu")

    # Extract classes names
    names = model.names

    # Video setup
    videocapture = cv2.VideoCapture(source)
    frame_width, frame_height = int(videocapture.get(3)), int(videocapture.get(4))
    fps, fourcc = int(videocapture.get(5)), cv2.VideoWriter_fourcc(*"mp4v")

    # Output setup
    video_writer = cv2.VideoWriter("output/object_counting_in_region.avi", fourcc, fps, (frame_width, frame_height))

    # Iterate over video frames
    while videocapture.isOpened():
        success, frame = videocapture.read()
        if not success:
            break
        vid_frame_count += 1

        # Extract the results
        results = model.track(frame, persist=True, classes=classes)

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            clss = results[0].boxes.cls.cpu().tolist()

            annotator = Annotator(frame, line_width=2, example=names)

            for box, track_id, cls in zip(boxes, track_ids, clss):
                annotator.box_label(box, label=names[cls], color=colors(cls, True))
                bbox_center = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2

                track = track_history[track_id]
                track.append((float(bbox_center[0]), float(bbox_center[1])))
                if len(track) > 30:
                    track.pop(0)
                points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(frame, [points], isClosed=False, color=colors(cls, True))

                # Check if detection inside region
                for region in counting_regions:
                    if region["coordinates"].contains(Point((bbox_center[0], bbox_center[1]))):
                        region["counts"] += 1

        # Draw regions
        for region in counting_regions:
            region_label = str(region["counts"])
            region_color = region["region_color"]
            region_text_color = region["text_color"]

            polygon_coords = np.array(region["coordinates"].exterior.coords, dtype=np.int32)
            centroid_x, centroid_y = int(region["coordinates"].centroid.x), int(region["coordinates"].centroid.y)

            text_size, _ = cv2.getTextSize(region_label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            text_x = centroid_x - text_size[0] // 2
            text_y = centroid_y + text_size[1] // 2
            cv2.rectangle(
                frame,
                (text_x - 5, text_y - text_size[1] - 5),
                (text_x + text_size[0] + 5, text_y + 5),
                region_color,
                -1,
            )
            cv2.putText(frame, region_label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, region_text_color)
            cv2.polylines(frame, [polygon_coords], True, region_color)

        if view_result:
            if vid_frame_count == 1:
                cv2.namedWindow("Region Counter Movable")
                cv2.setMouseCallback("Region Counter Movable", mouse_callback)
            cv2.imshow("Region Counter Movable", frame)

        if save_result:
            video_writer.write(frame)

        for region in counting_regions:  # Reinitialize count for each region
            region["counts"] = 0

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    del vid_frame_count
    video_writer.release()
    videocapture.release()
    cv2.destroyAllWindows()


def parse_opt():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default="yolov8n.pt", help="initial weights path")
    parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--source", type=str, required=True, help="video file path")
    parser.add_argument("--view_result", action="store_true", help="show results")
    parser.add_argument("--save_result", action="store_true", help="save results")
    parser.add_argument("--classes", nargs="+", type=int, help="filter by class: --classes 0, or --classes 0 2 3")

    return parser.parse_args()


def main(opt):
    """Main function."""
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)