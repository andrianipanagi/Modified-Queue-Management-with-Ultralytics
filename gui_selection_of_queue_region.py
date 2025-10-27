# import libraries
import cv2
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np

# create a GUI to upload a frame from a video and select 4 points to define a queue region
class QueueRegionSelector:
    def __init__(self, root):
        self.root = root
        self.root.title("Queue Region Selector")

        self.video_path = None
        self.points = []
        self.frame = None
        self.canvas_frame = None
        self.canvas = None
        self.scrollable_canvas = None
        self.scrollbar = None
        self.image_tk = None
        self.point_labels = []  # To store the point labels on the canvas
        self.original_frame_size = None # Store original frame dimensions

        # Instructions Label
        self.instructions_label = tk.Label(root, text="1. Upload a video.\n2. Click on the image to select 4 points for the queue region.\n   Click points in this order: Top-Left, Top-Right, Bottom-Right, Bottom-Left.\n3. Click 'Create Region'.", justify="left")
        self.instructions_label.pack(pady=10)

        # Upload Button
        self.upload_button = tk.Button(root, text="Upload Video", command=self.upload_video)
        self.upload_button.pack(pady=10)

        # Create Region Button
        self.create_region_button = tk.Button(root, text="Create Region", command=self.create_region, state=tk.DISABLED)
        self.create_region_button.pack(pady=10)

        # Status Label
        self.status_label = tk.Label(root, text="No video uploaded")
        self.status_label.pack(pady=10)

    def upload_video(self):
        """
        Opens a file dialog to select a video file, displays the first frame,
        and enables the create region button.
        """
        self.video_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4;*.avi;*.mov")])
        if self.video_path:
            self.status_label.config(text=f"Video uploaded: {self.video_path}")
            cap = cv2.VideoCapture(self.video_path)
            ret, self.frame = cap.read()
            cap.release()
            if ret:
                self.original_frame_size = self.frame.shape[:2] # Store original dimensions
                self.display_frame()
                self.create_region_button.config(state=tk.NORMAL)  # Enable the button
                self.points = []  # Reset points
                self.point_labels = [] # Clear any existing labels
            else:
                self.status_label.config(text="Error: Could not read the video file.")
        else:
            self.status_label.config(text="No video uploaded")

    
    def display_frame(self):
        if self.frame is None:
            return

        # Use the original frame without resizing
        height, width = self.original_frame_size
        frame_rgb = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        self.image_tk = ImageTk.PhotoImage(img)

        # Create scrollable canvas once
        if self.scrollable_canvas is None:
            self.canvas_frame = tk.Frame(self.root)
            self.canvas_frame.pack(fill=tk.BOTH, expand=True)

            self.scrollable_canvas = tk.Canvas(self.canvas_frame)
            self.scrollable_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

            self.scrollbar = tk.Scrollbar(self.canvas_frame, orient=tk.VERTICAL, command=self.scrollable_canvas.yview)
            self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

            self.scrollable_canvas.configure(yscrollcommand=self.scrollbar.set)
            self.scrollable_canvas.bind('<Configure>', lambda e: self.scrollable_canvas.configure(scrollregion=self.scrollable_canvas.bbox("all")))

            # Embed an inner frame inside the scrollable canvas
            self.canvas_container = tk.Frame(self.scrollable_canvas)
            self.scrollable_canvas.create_window((0, 0), window=self.canvas_container, anchor='nw')

            # Create actual canvas inside container to draw image
            self.canvas = tk.Canvas(self.canvas_container, width=width, height=height)
            self.canvas.pack()
            self.canvas.bind("<Button-1>", self.on_canvas_click)
            self.canvas_image = self.canvas.create_image(0, 0, anchor='nw', image=self.image_tk)

        else:
            self.canvas.config(width=width, height=height)
            self.canvas.itemconfig(self.canvas_image, image=self.image_tk)
            self.canvas.delete("point")
            for label in self.point_labels:
                label.destroy()
            self.point_labels = []


    def on_canvas_click(self, event):
        """
        Handles mouse clicks on the canvas to select points for the queue region.
        """
        if len(self.points) < 4:
            x, y = event.x, event.y
            self.points.append((x, y))
            point_number = len(self.points)
            self.canvas.create_oval(x - 5, y - 5, x + 5, y + 5, fill="red", tags="point")  # Mark the point
            label = tk.Label(self.canvas, text=str(point_number), bg="red", fg="white")
            label.place(x=x, y=y)
            self.point_labels.append(label) #store the label so we can destroy it later
            self.status_label.config(text=f"Point {point_number} selected: ({x}, {y})")
            if len(self.points) == 4:
                self.status_label.config(text="Four points selected. Click 'Create Region'.")

    def create_region(self):
        """
        Creates the queue region and displays the coordinates.
        Scales the points back to the original frame size.
        """
        if len(self.points) != 4:
            self.status_label.config(text="Error: Please select exactly four points.")
            return

        self.status_label.config(text=f"Queue region created with points: {self.points}")

        #print at the end the points
        print("Queue region points:", self.points)

        # Disable the button after region creation.
        self.create_region_button.config(state=tk.DISABLED)
        self.upload_button.config(state=tk.DISABLED)

    def get_points(self):
        return self.points


root = tk.Tk()
app = QueueRegionSelector(root)
root.mainloop()

if __name__ == "__main__":
    points = app.get_points()
    if points:
        print("Selected queue region points:", points)
    else:
        print("No points selected.")