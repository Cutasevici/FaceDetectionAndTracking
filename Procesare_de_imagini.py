import tkinter as tk
import cv2
import numpy as np
import time
import datetime
import win32gui
import win32ui
import win32con
import imutils
import threading
import face_recognition
import os
from PIL import ImageGrab

#for experiments
current_face_id = 1
face_id_mapping = {}


# Initialize screen capture resources
start_cycle = True
last_face_check_time = 0
# Global variable for background subtraction model
reinitialize=False
background_model = None
stop_recording_event = threading.Event()
hwin = win32gui.GetDesktopWindow()
hwindc = win32gui.GetWindowDC(hwin)
srcdc = win32ui.CreateDCFromHandle(hwindc)
memdc = srcdc.CreateCompatibleDC()
bmp = win32ui.CreateBitmap()
motion_start_time = None
is_recording = False
trackers = []
use_tracking = False
face_locations =[]
# Define the region of the screen to capture and other settings
capture_x, capture_y, width, height = 100, 100, 800, 400
blue_border_thickness = 4
red_border_thickness = 2
blue_offset = 2
red_offset = 4
frame_rate = 60
frame_interval = int(1000 / frame_rate)
last_frame_time = time.time()
global_trackers = []
# FPS related variables
frame_count = 0
fps_accumulated = 0
fps_display = 0


def detect_faces(frame):
    global frame_count, global_trackers, start_cycle, cycle_finished

    if not start_cycle:
        return []
    print("Detect faces was called")


    cycle_length = 205  # Total cycle length: 5 frames for detection + 100 frames cooldown
    detection_frames = 100  # Number of frames for face detection
    face_locations = []
    print("call to detect_faces")
    # Validate input frame
    if frame is None or frame.size == 0:
        print("Invalid or empty frame provided to detect_faces")
        return face_locations

    cycle_position = frame_count % cycle_length

    if cycle_position < detection_frames:
        try:
            # Perform face detection using face_recognition library
            rgb_frame = frame[:, :, ::-1]  # Convert BGR (OpenCV format) to RGB
            detected_faces = face_recognition.face_locations(rgb_frame)

            # Initialize trackers for the detected faces
            for top, right, bottom, left in detected_faces:
                # Extend the face box by 20% for better tracking
                height = bottom - top
                width = right - left
                top = max(int(top - height * 0.2), 0)
                bottom = min(int(bottom + height * 0.2), frame.shape[0])
                left = max(int(left - width * 0.2), 0)
                right = min(int(right + width * 0.2), frame.shape[1])

                x, y, w, h = left, top, right - left, bottom - top

                # Create a new tracker for the extended face area and add it to the list
                tracker = cv2.TrackerKCF_create()
                tracker.init(frame, (x, y, w, h))
                global_trackers.append(tracker)



                face_locations.append((x, y, w, h))

        except Exception as e:
            print(f"Error during face detection: {e}")

    frame_count += 1


    return face_locations  # Return the detected face locations if any, otherwise an empty list






def track_faces(frame, face_locations, reinitialize=False):
    # Use default dictionaries to store current ID and tracker-ID mapping
    current_face_id = track_faces.current_face_id if hasattr(track_faces, 'current_face_id') else 1
    tracker_to_id = track_faces.tracker_to_id if hasattr(track_faces, 'tracker_to_id') else {}
    max_id = 100

    # Initialize or retrieve the global 'trackers' list
    global trackers
    trackers = trackers if 'trackers' in globals() else []

    tracked_faces_with_ids = []

    # Downsampling for resource optimization
    scale_factor = 0.5
    small_frame = cv2.resize(frame, (0, 0), fx=scale_factor, fy=scale_factor)

    # If reinitializing, clear existing trackers and reset mappings
    if reinitialize:
        trackers.clear()
        tracker_to_id.clear()

        # Restrict the number of trackers for optimization
        for (x, y, w, h) in face_locations[:10]:  # Limit to 10 trackers
            tracker = cv2.TrackerCSRT_create()
            scaled_face = (int(x * scale_factor), int(y * scale_factor), int(w * scale_factor), int(h * scale_factor))
            tracker.init(small_frame, scaled_face)
            trackers.append(tracker)
            tracker_to_id[tracker] = current_face_id
            current_face_id = (current_face_id % max_id) + 1

    # Update existing trackers
    to_remove = []
    for tracker in trackers:
        success, box = tracker.update(small_frame)
        if success:
            x, y, w, h = [int(v / scale_factor) for v in box]  # Scale back up to original frame size
            face_id = tracker_to_id.get(tracker, 0)
            tracked_faces_with_ids.append(((x, y, x + w, y + h), face_id))
        else:
            print("Tracking failed")
            to_remove.append(tracker)

    # Remove trackers for objects that are no longer present
    for tracker in to_remove:
        trackers.remove(tracker)
        del tracker_to_id[tracker]

    # Update the global variables
    track_faces.current_face_id = current_face_id
    track_faces.tracker_to_id = tracker_to_id
    return tracked_faces_with_ids






def capture_screen(region):
    try:
        if region[2] <= 0 or region[3] <= 0:
            print("Invalid capture region dimensions.")
            return None

        bmp.CreateCompatibleBitmap(srcdc, region[2], region[3])
        memdc.SelectObject(bmp)
        memdc.BitBlt((0, 0), (region[2], region[3]), srcdc, (region[0], region[1]), win32con.SRCCOPY)

        bmp_info = bmp.GetInfo()
        bmp_str = bmp.GetBitmapBits(True)
        img = np.frombuffer(bmp_str, dtype='uint8')
        img.shape = (bmp_info['bmHeight'], bmp_info['bmWidth'], 4)

        return img

    except Exception as e:
        print(f"Error capturing screen: {e}")
        return None


def draw_tracked_faces(frame, tracked_faces_with_ids):
    for (face_location, face_id) in tracked_faces_with_ids:
        left, top, right, bottom = face_location
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, f'ID: {face_id}', (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)





def update():
    global last_frame_time, frame_count, fps_accumulated, \
        fps_display, is_recording, motion_start_time, \
        continuous_face_detection, face_locations, use_tracking, trackers,\
        start_cycle, cycle_finished,\
        last_detection_time, last_face_check_time, current_face_id,\
        face_id_mapping, tracked_faces_with_ids,known_face_encodings,known_face_identities

    try:
        current_time = time.time()
        screenshot_region = (
            capture_x + red_offset, capture_y + red_offset, width - 2 * red_offset, height - 2 * red_offset)
        screen = capture_screen(screenshot_region)

        if screen is None:
            raise Exception("Failed to capture screen.")
        # Convert to writable format
        writable_screen = np.copy(screen)
        frame = cv2.cvtColor(writable_screen, cv2.COLOR_BGRA2BGR)

        motion_duration = detect_motion(frame)

        global last_detection_time, start_cycle, cycle_finished

        # Initialize last_detection_time to None at the start of your program
        last_detection_time = None

        if motion_duration > 3:
            if not is_recording:
                is_recording = True
                threading.Thread(target=lambda: start_window_recording(root, "E:/projectVideoOutput")).start()
                use_tracking = True


            if current_time - last_face_check_time > 10:
                last_face_check_time = current_time
                face_locations = detect_faces(frame)
                if face_locations:
                    if face_locations:
                        # Reinitialize trackers with new face locations
                        tracked_faces = track_faces(frame, face_locations, reinitialize=True)
                        # Stop further checks if a face is found
                        last_face_check_time = float('inf')

            # Add a delay to reduce resource usage



        elif motion_duration > 0:
            cv2.putText(frame, "Motion Detected", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)



        tracked_faces_with_ids = track_faces(frame, face_locations)
        draw_tracked_faces(frame, tracked_faces_with_ids)



        # Calculate FPS
        time_diff = current_time - last_frame_time
        fps_accumulated += 1 / time_diff if last_frame_time != 0 else 0
        frame_count += 1

        if frame_count >= frame_rate:
            fps_display = fps_accumulated / frame_count
            fps_accumulated = 0
            frame_count = 0

        if fps_display > 0:
            cv2.putText(frame, f"FPS: {fps_display:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        last_frame_time = current_time
        cv2.imshow("Region Capture", frame)

        if cv2.waitKey(1) == ord('q'):
            if is_recording:
                print("Waiting for recording to finish...")
                stop_recording_event.set()
                is_recording = False

            start_cycle = False
            cycle_finished = True
            cleanup()
            root.destroy()

    except Exception as e:
        print(f"An error occurred in update function: {e}")

    root.after(frame_interval, update)



def cleanup():
    srcdc.DeleteDC()
    memdc.DeleteDC()
    win32gui.ReleaseDC(hwin, hwindc)
    win32gui.DeleteObject(bmp.GetHandle())
    cv2.destroyAllWindows()

def detect_motion(frame):
    global motion_start_time, background_model

    # Initialize background model if it doesn't exist
    if background_model is None:
        background_model = cv2.createBackgroundSubtractorMOG2(history=1000, varThreshold=25, detectShadows=True)

    # Apply a blur to reduce noise and small changes
    blurred_frame = cv2.GaussianBlur(frame, (5, 5), 0)
    resized_frame = imutils.resize(blurred_frame, width=500)

    # Apply the background model to get foreground mask
    fg_mask = background_model.apply(resized_frame)
    thresh = cv2.threshold(fg_mask, 25, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=3)  # Increased dilation

    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    motion_detected = False
    for contour in contours:
        if cv2.contourArea(contour) > 1000:  # Increased contour area threshold
            motion_detected = True
            break

    if motion_detected:
        if motion_start_time is None:
            motion_start_time = time.time()
        return time.time() - motion_start_time
    else:
        motion_start_time = None
        return 0



def start_window_recording(window, output_directory="E:\\projectVideoOutput", base_filename="MotionVideo"):
    global is_recording, stop_recording_event

    if not is_recording:
        return

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{base_filename}_{timestamp}.mp4"
    full_output_path = os.path.join(output_directory, filename)
    print(f"Starting window recording: {full_output_path}")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(full_output_path, fourcc, frame_rate, (width, height))

    start_time = time.time()
    while not stop_recording_event.is_set() and (time.time() - start_time) < 60:
        captured_frame = capture_tkinter_window(window)
        if captured_frame is None:
            continue

        frame_bgr = cv2.cvtColor(np.array(captured_frame), cv2.COLOR_RGB2BGR)

        # Track faces in the frame using track_faces
        tracked_faces_with_ids = track_faces(frame_bgr, face_locations)
        draw_tracked_faces(frame_bgr, tracked_faces_with_ids)

        out.write(frame_bgr)
        time.sleep(0.1)

    out.release()
    print(f"Window recording finished: {full_output_path}")
    is_recording = False


def capture_tkinter_window(window):
    x0 = window.winfo_rootx()
    y0 = window.winfo_rooty()
    x1 = x0 + window.winfo_width()
    y1 = y0 + window.winfo_height()
    return ImageGrab.grab(bbox=(x0, y0, x1, y1))

root = tk.Tk()
root.overrideredirect(True)
root.geometry(f"{width}x{height}+{capture_x}+{capture_y}")
root.lift()
root.wm_attributes("-topmost", True)
root.wm_attributes("-transparentcolor", "white")

canvas = tk.Canvas(root, bg='white', highlightthickness=0)
canvas.pack(fill=tk.BOTH, expand=True)
canvas.create_rectangle(blue_offset, blue_offset, width - blue_offset, height - blue_offset, outline='blue',
                        width=blue_border_thickness)
inner_red_offset = blue_offset + red_offset
canvas.create_rectangle(inner_red_offset, inner_red_offset, width - inner_red_offset, height - inner_red_offset,
                        outline='red', width=red_border_thickness)

root.after(frame_interval, update)
root.mainloop()
