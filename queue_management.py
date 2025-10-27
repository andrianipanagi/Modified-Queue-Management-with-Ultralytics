import cv2
import numpy as np
from ultralytics import solutions
from ultralytics.solutions.solutions import BaseSolution, SolutionAnnotator, SolutionResults
from ultralytics.utils.plotting import colors

class QueueManager(BaseSolution):
    """
    Manages queue counting in real-time video streams based on object tracks.

    This class extends BaseSolution to provide functionality for tracking and counting objects within a specified
    region in video frames.

    Attributes:
        counts (int): The current count of objects in the queue.
        rect_color (Tuple[int, int, int]): RGB color tuple for drawing the queue region rectangle.
        region_length (int): The number of points defining the queue region.
        track_line (List[Tuple[int, int]]): List of track line coordinates.
        track_history (Dict[int, List[Tuple[int, int]]]): Dictionary storing tracking history for each object.

    Methods:
        initialize_region: Initializes the queue region.
        process: Processes a single frame for queue management.
        extract_tracks: Extracts object tracks from the current frame.
        store_tracking_history: Stores the tracking history for an object.
        display_output: Displays the processed output.

    Examples:
        >>> cap = cv2.VideoCapture("path/to/video.mp4")
        >>> queue_manager = QueueManager(region=[100, 100, 200, 200, 300, 300])
        >>> while cap.isOpened():
        >>>     success, im0 = cap.read()
        >>>     if not success:
        >>>         break
        >>>     results = queue_manager.process(im0)
    """

    def __init__(self, **kwargs):
        """Initializes the QueueManager with parameters for tracking and counting objects in a video stream."""
        super().__init__(**kwargs)
        self.initialize_region()
        self.counts = 0  # Queue counts information
        self.rect_color = (255, 255, 255)  # Rectangle color for visualization
        self._display_counts = True  # Flag to display counts on the video
        self.region_length = len(self.region)  # Store region length for further usage
        
    def hide_counts(self):
        """Hides the queue counts display."""
        self._display_counts = False

    def show_counts(self):
        """Shows the queue counts display."""
        self._display_counts = True

    def process(self, im0):
        """
        Process queue management for a single frame of video.

        Args:
            im0 (numpy.ndarray): Input image for processing, typically a frame from a video stream.

        Returns:
            (SolutionResults): Contains processed image `im0`, 'queue_count' (int, number of objects in the queue) and
                'total_tracks' (int, total number of tracked objects).

        Examples:
            >>> queue_manager = QueueManager()
            >>> frame = cv2.imread("frame.jpg")
            >>> results = queue_manager.process(frame)
        """
        self.counts = 0  # Reset counts every frame
        self.extract_tracks(im0)  # Extract tracks from the current frame
        annotator = SolutionAnnotator(im0, line_width=self.line_width)  # Initialize annotator
        annotator.draw_region(reg_pts=self.region, color=self.rect_color, thickness=self.line_width * 2)  # Draw region
        
        # Filter objects based on their position relative to the queue region
        region_poly = np.array(self.region, np.int32)
        filtered_data = []
        # Iterate over all detected and tracked objects
        for box, track_id, cls, conf in zip(self.boxes, self.track_ids, self.clss, self.confs):
            x1, y1, x2, y2 = [int(x) for x in box]
            
            # Use the bottom-center point of the bounding box as the representative point for the object
            center_x = (x1 + x2) // 2
            center_y = (y1+y2) //2  #since is the bottom where the person feet are
            
            # Check if the representative point is inside the queue region polygon
            # cv2.pointPolygonTest returns > 0 if the point is inside, 0 if on the edge, and < 0 if outside.
            is_inside = cv2.pointPolygonTest(region_poly, (center_x, center_y), False) >= 0 
            
            if is_inside:
                # Keep the data only for objects inside the region
                filtered_data.append((box, track_id, cls, conf))

        self.boxes, self.track_ids, self.clss, self.confs = zip(*filtered_data) if filtered_data else ([], [], [], [])

        for box, track_id, cls, conf in zip(self.boxes, self.track_ids, self.clss, self.confs):
            # Draw bounding box and counting region
            annotator.box_label(box, label=self.adjust_box_label(cls, conf, track_id), color=colors(track_id, True))
            self.store_tracking_history(track_id, box)  # Store track history

            # Cache frequently accessed attributes
            track_history = self.track_history.get(track_id, [])

            # Store previous position of track and check if the object is inside the counting region
            prev_position = None
            if len(track_history) > 1:
                prev_position = track_history[-2]
            if self.region_length >= 3 and prev_position and self.r_s.contains(self.Point(self.track_line[-1])):
                self.counts += 1

        # Display queue counts
        if self._display_counts:
            annotator.queue_counts_display(
                f"Queue Counts : {str(self.counts)}",
                points=self.region,
                region_color=self.rect_color,
                txt_color=(104, 31, 17),
            )
        plot_im = annotator.result()
        self.display_output(plot_im)  # Display output with base class function

        # Return a SolutionResults object with processed data
        return SolutionResults(plot_im=plot_im, queue_count=self.counts, total_tracks=len(self.track_ids))
    
def process_video(video_path="example_video.mp4", output_path="output.mp4"):
    """
    Processes a video file to manage queue counting and alerts based on object tracking.

    This function opens a video file, initializes the QueueManager, and processes each frame to count objects
    in a specified queue region. It also implements congestion and dwell time alerts.

    The processed video is saved to an output file and displayed in real-time.

    Examples:
        >>> process_video()
    """
    # open video
    cap = cv2.VideoCapture(video_path) 
    assert cap.isOpened(), "Error opening video file"

    # video properties
    w,h, fps = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), cap.get(cv2.CAP_PROP_FPS)

    # queue region from qui
    queue_region =  [(217, 288), (342, 436), (562, 225), (455, 147)]

    # video writer
    video_writer = cv2.VideoWriter(output_path,cv2.VideoWriter_fourcc(*"mp4v"),fps,(w,h))

    # use here the QueueManager class from code modified from above
    queue = QueueManager(
        model="yolo11n.pt", 
        classes_names = ["person"],
        region=queue_region,
    )

    # hide the text box count on each frame 
    queue.hide_counts()

    # congestion alert parameters
    CONGESTION_THRESHOLD = 3 
    ALERT_MESSAGE = "❗️ CONGESTION ALERT! Queue too long."

    # person stands over 5 seconds logic
    person_dwell_times = {} 
    frame_number = 0 
    DWELL_TIME_SECONDS = 5
    fps = cap.get(cv2.CAP_PROP_FPS) # Ensure fps is correct
    DWELL_TIME_FRAMES = int(DWELL_TIME_SECONDS * fps)

    while cap.isOpened():
        success, im0 = cap.read()
        if not success:
            break
        
        results = queue.process(im0)
        
        #extract the annotated frame from the SolutionResults object, this is the numpy array of the processed frame
        annotated_frame = results.plot_im
        
        current_person_ids = set()

        # These IDs represent people currently detected inside the queue region.
        for track_id in queue.track_ids: 
            current_person_ids.add(track_id)

            # Update/Add person's start frame if they are new to the queue
            if track_id not in person_dwell_times:
                person_dwell_times[track_id] = frame_number

            # Check for DWELL TIME ALERT
            if (frame_number - person_dwell_times[track_id]) >= DWELL_TIME_FRAMES:
                DWELL_ALERT_MESSAGE = f"TIME ALERT: Person {track_id} is waiting too long!"
                cv2.putText(annotated_frame, DWELL_ALERT_MESSAGE, (10, 130), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 0, 255), 2)


        # Remove IDs that have left the region/frame
        all_known_ids = list(person_dwell_times.keys())
        for track_id in all_known_ids:
            if track_id not in current_person_ids:
                del person_dwell_times[track_id]

        frame_number += 1

        if annotated_frame is not None and isinstance(annotated_frame, np.ndarray):
            queue_count = results.queue_count
            cv2.putText(annotated_frame, f"Queue Count: {queue_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # CONGESTION ALERT LOGIC
            if queue_count > CONGESTION_THRESHOLD:
                # Display the alert in red
                cv2.putText(annotated_frame, ALERT_MESSAGE, (10, 80), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 255), 2)

            video_writer.write(annotated_frame)

            #Display the processed video in real-time
            cv2.imshow("Processed Video", annotated_frame)
            
        else:
            print("Error: Annotated frame not found in results or is not a valid NumPy array.")
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        

    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    process_video()