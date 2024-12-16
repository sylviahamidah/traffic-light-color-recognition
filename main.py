import cv2
import numpy as np
import argparse
import json

def load_rois(config_path, video_name):
    # Load ROIs from the JSON file based on the video name
    with open(config_path, 'r') as f:
        roi_config = json.load(f)
    return roi_config.get(video_name, [])

def detect_color(hsv_frame):
    # Define HSV ranges for red, yellow, and green
    red_lower1 = np.array([0, 100, 100])
    red_upper1 = np.array([10, 255, 255])
    red_lower2 = np.array([160, 100, 100])
    red_upper2 = np.array([180, 255, 255])

    yellow_lower = np.array([10, 50, 50])
    yellow_upper = np.array([35, 255, 255])

    green_lower = np.array([35, 50, 50])
    green_upper = np.array([85, 255, 255])

    # Create masks for each color
    red_mask1 = cv2.inRange(hsv_frame, red_lower1, red_upper1)
    red_mask2 = cv2.inRange(hsv_frame, red_lower2, red_upper2)
    red_mask = red_mask1 | red_mask2

    yellow_mask = cv2.inRange(hsv_frame, yellow_lower, yellow_upper)
    green_mask = cv2.inRange(hsv_frame, green_lower, green_upper)

    return red_mask, yellow_mask, green_mask


def process_video(video_path, output_path, log_path, threshold):
    cap = cv2.VideoCapture(video_path)

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))


    # Open log file for writing
    with open(log_path, "w") as log_file:
        log_file.write("Time (s),Color,ROI Coordinates (x,y,w,h)\n")  # Write header

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Calculate current frame time
            frame_index = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            frame_time = frame_index / fps

            hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            # Process each ROI
            for roi in rois:
                x, y, w, h = roi
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)  # White rectangle
                roi_frame = frame[y:y+h, x:x+w]
                roi_hsv_frame = hsv_frame[y:y+h, x:x+w]

                # Detect colors in the ROI
                red_mask, yellow_mask, green_mask = detect_color(roi_hsv_frame)

                # Find contours for each color
                for color, mask, color_box in zip(
                    ["Red", "Yellow", "Green"],
                    [red_mask, yellow_mask, green_mask],
                    [(0, 0, 255), (0, 255, 255), (0, 255, 0)],
                ):
                    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

                    for contour in contours:
                        # Filter small contours to reduce false positives
                        if cv2.contourArea(contour) > threshold:
                            x_c, y_c, w_c, h_c = cv2.boundingRect(contour)
                            aspect_ratio = float(w_c) / h_c
                            if 0.8 <= aspect_ratio <= 1.2:  # Filter shapes close to square
                                # Draw rectangle around the detected area
                                cv2.rectangle(roi_frame, (x_c, y_c), (x_c + w_c, y_c + h_c), (color_box), 2)

                                # Calculate text size
                                text_size, baseline = cv2.getTextSize(color, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                                text_width, text_height = text_size
                                padding = 5

                                # Define the rectangle for the background
                                rect_start = (x_c, y_c - text_height - padding - 10)
                                rect_end = (x_c + text_width + 2 * padding, y_c - 10)

                                # Draw the background rectangle
                                cv2.rectangle(roi_frame, rect_start, rect_end, color_box, -1)

                                # Draw the text over the rectangle
                                cv2.putText(roi_frame, color, (x_c + padding, y_c - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

                                # Write to log file
                                log_file.write(f"{frame_time:.2f},{color},{(x, y, w, h)}\n")

                # Overlay the processed ROI back onto the frame
                frame[y:y+h, x:x+w] = roi_frame

            # Write the frame to the output video
            out.write(frame)

            # Show the processed frame
            cv2.imshow("Traffic Light Detection", frame)

            # Exit on pressing 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Traffic Light Detection with Manual ROIs")
    parser.add_argument("video", help="Path to the input video")
    parser.add_argument("output", help="Path to save the output video")
    parser.add_argument("--log", help="Path to save the log file", default="log_output.txt")

    args = parser.parse_args()

    # Load ROIs from the configuration file
    rois = load_rois("roi.json", args.video)

    if not rois:
        print(f"No ROIs defined for {args.video} in the configuration file.")
    else:
        if args.video == 'sample_1.mp4' or args.video == 'sample_3.mp4':
            threshold = 500
        elif args.video == 'sample_2.mp4':
            threshold = 30
        process_video(args.video, args.output, args.log, threshold)
        