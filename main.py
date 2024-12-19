import tkinter as tk
from tkinter import ttk
from tkinter import filedialog, messagebox, Scale
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image, ImageTk, ImageFilter
import cv2
import numpy as np

# Bayer Matrix for Dithering
def bayer_matrix(n):
        if n == 0:
            return [
                [0, 2],
                [3, 1]
            ]
        else:
            prev_matrix = bayer_matrix(n - 1)
            size = len(prev_matrix)
            new_matrix = [[0] * (2 * size) for _ in range(2 * size)]

            for i in range(size):
                for j in range(size):
                    new_matrix[i][j] = 4 * prev_matrix[i][j]
                    new_matrix[i][j + size] = 4 * prev_matrix[i][j] + 2
                    new_matrix[i + size][j] = 4 * prev_matrix[i][j] + 3
                    new_matrix[i + size][j + size] = 4 * prev_matrix[i][j] + 1

            return new_matrix

class ImageProcessingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Processing App")
        self.root.resizable(False, False)
        self.image = None
        self.processed_image = None
        self.threshold = 100  # Default binary threshold
        self.current_mode = None  # Track the current mode
        self.original_image = None  # Keep a copy of the original image
        self.original_height = 0
        self.original_width = 0
        self.dx = 0
        self.dy = 0

        # UI Elements
        self.canvas_frame = tk.Frame(root)
        self.canvas = tk.Canvas(self.canvas_frame, width=850, height=550, bg='gray')
        self.canvas.pack()
        self.canvas_frame.grid(row=1, column=1)

        self.control_frame = tk.Frame(root)

        # Button Row (Load, Save)
        self.btn_row = tk.Frame(self.control_frame)
        self.btn_load = tk.Button(self.btn_row, text="Load Image", command=self.load_image)
        self.btn_load.pack(side=tk.LEFT, padx=5, pady=5)

        self.btn_reset = tk.Button(self.btn_row, text="Reset", command=self.init_image, state=tk.DISABLED)
        self.btn_reset.pack(side=tk.LEFT, padx=5, pady=5)

        self.btn_rotate = tk.Button(self.btn_row, text="Rotate 90°", command=self.rotate_image, state=tk.DISABLED)
        self.btn_rotate.pack(side=tk.LEFT, padx=5, pady=5)

        self.btn_resize = tk.Button(self.btn_row, text="Resize", command=self.open_resize_window, state=tk.DISABLED)
        self.btn_resize.pack(side=tk.LEFT, padx=5, pady=5)

        self.btn_translate = tk.Button(self.btn_row, text="Translate", command=self.open_translate_window, state=tk.DISABLED)
        self.btn_translate.pack(side=tk.LEFT, padx=5, pady=5)

        self.btn_save = tk.Button(self.btn_row, text="Save Image", command=self.save_image, state=tk.DISABLED)
        self.btn_save.pack(side=tk.LEFT, padx=5, pady=5)
        self.btn_row.grid(row=1, column=1)

        # Combobox Row
        self.combo_row = tk.Frame(self.control_frame)
        self.color_label = tk.Label(self.combo_row, text="Color")
        self.color_label.pack(side=tk.LEFT, padx=5)
        color_list = ["Original Colors", "Negative (CMY)", "Grayscale", "Inverse Grayscale", "Binary", "Inverse Binary", "HSV", "LAB"]
        self.mode_combobox = ttk.Combobox(self.combo_row, values=color_list, state=tk.DISABLED)
        self.mode_combobox.pack(side=tk.LEFT, padx=5, pady=5)
        self.mode_combobox.bind("<<ComboboxSelected>>", self.apply_mode)

        self.effect_label = tk.Label(self.combo_row, text="Effect")
        self.effect_label.pack(side=tk.LEFT, padx=5)
        effect_list = ["None", "Quantization", "Dithering", "Median Cut"]
        self.effect_combobox = ttk.Combobox(self.combo_row, values=effect_list, state=tk.DISABLED)
        self.effect_combobox.pack(side=tk.LEFT, padx=5, pady=5)
        self.effect_combobox.bind("<<ComboboxSelected>>", self.enable_effect)

        quantization_list = ["32-Bit", "16-Bit", "8-Bit", "4-Bit"]
        self.quantization_combobox = ttk.Combobox(self.combo_row, values=quantization_list)
        self.quantization_combobox.bind("<<ComboboxSelected>>", self.apply_quantization)

        dithering_list = ["2x2", "3x3", "4x4", "8x8", "16x16", "32x32", "64x64"]
        self.dithering_combobox = ttk.Combobox(self.combo_row, values=dithering_list)
        self.dithering_combobox.bind("<<ComboboxSelected>>", self.apply_dithering)
        self.combo_row.grid(row=2, column=1)

        # Effect Row (Threshold)
        self.effect_row = tk.Frame(self.control_frame)
        self.threshold_slider = Scale(self.effect_row, from_=0, to=255, orient=tk.HORIZONTAL, label="Binary Threshold", length=200, state=tk.DISABLED)
        self.threshold_slider.set(100)
        self.threshold_slider.bind("<ButtonRelease-1>", self.update_binary_image)

        self.init_label = tk.Label(self.effect_row, text="Load an image from your device using the \"Load Image\" button.")
        self.init_label.pack(side=tk.LEFT, padx=5, pady=19)
        self.effect_row.grid(row=3, column=1)

        # Histogram Button Row
        self.hist_row = tk.Frame(self.control_frame)
        self.btn_hist_gray = tk.Button(self.hist_row, text="Show Grayscale Histogram", command=self.show_gray_hist, state=tk.DISABLED)
        self.btn_hist_gray.pack(side=tk.LEFT, padx=5, pady=5)

        self.btn_hist_color = tk.Button(self.hist_row, text="Show RGB Histogram", command=self.show_color_hist, state=tk.DISABLED)
        self.btn_hist_color.pack(side=tk.LEFT, padx=5, pady=5)
        self.hist_row.grid(row=4, column=1)

        self.control_frame.grid(row=2, column=1)

    def load_image(self):
        filepath = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.png;*.jpeg")])
        if not filepath:
            return
        self.loaded_image = cv2.imread(filepath)
        self.loaded_image = cv2.cvtColor(self.loaded_image, cv2.COLOR_BGR2RGB)
        self.init_image()
        self.update_control_state()
        self.original_height = self.loaded_image.shape[0]
        self.original_width = self.loaded_image.shape[1]
        self.init_label.pack_forget()
    
    def init_image(self):
        self.image = self.processed_image = self.original_image = self.loaded_image # Keep a copy of the original image
        self.refresh_image()
        self.dx = 0
        self.dy = 0
        self.mode_combobox.set('Original Colors')
        self.effect_combobox.set('None')
        self.quantization_combobox.set('32-Bit')
        self.dithering_combobox.set('2x2')
        self.threshold_slider.pack_forget()

    def refresh_image(self):
        if self.processed_image is None:
            return

        # Ensure image dimensions are within canvas limits
        height = self.image.shape[0]
        width = self.image.shape[1]
        canvas_width, canvas_height = self.canvas.winfo_width(), self.canvas.winfo_height()
        # Initialize resized_image to the original processed_image by default
        resized_image = self.processed_image
        if width > canvas_width or height > canvas_height:
            scale_factor = min(canvas_width / width, canvas_height / height)
            resized_image = cv2.resize(self.processed_image, (int(width * scale_factor), int(height * scale_factor)))

        img_tk = ImageTk.PhotoImage(Image.fromarray(resized_image))
        self.canvas.image = img_tk  # Keep a reference to avoid garbage collection
        self.canvas.create_image(canvas_width // 2, canvas_height // 2, image=img_tk)

    def update_control_state(self):
        # Enable controls only if an image is loaded
        state = tk.NORMAL if self.image is not None else tk.DISABLED
        self.mode_combobox.config(state=state)
        self.threshold_slider.config(state=state)
        self.btn_save.config(state=state)
        self.btn_hist_color.config(state=state)
        self.btn_hist_gray.config(state=state)
        self.btn_rotate.config(state=state)
        self.btn_resize.config(state=state)
        self.btn_reset.config(state=state)
        self.btn_translate.config(state=state)
        self.mode_combobox.config(state=state)
        self.effect_combobox.config(state=state)
    
    def rotate_image(self):
        if self.image is not None:
            self.original_image = cv2.rotate(self.original_image, cv2.ROTATE_90_CLOCKWISE)
            self.image = self.processed_image = cv2.rotate(self.processed_image, cv2.ROTATE_90_CLOCKWISE)
            self.refresh_image()
    
    def resize_image(self, new_height, new_width):
        self.image = self.processed_image = cv2.resize(self.processed_image, (new_width, new_height))
        self.refresh_image()

    def open_resize_window(self):
        def reset_size():
            width = self.original_width
            height = self.original_height
            self.resize_image(height, width)
            resize_window.destroy()  # Close the resize window

        def apply_resize():
            try:
                width = int(width_entry.get())
                height = int(height_entry.get())
                self.resize_image(width, height)
                resize_window.destroy()  # Close the resize window
            except ValueError:
                messagebox.showerror("Invalid Input", "Please enter valid integers for width and height.")

        def cancel_resize():
            resize_window.destroy()  # Close the resize window

        # Create the pop-up window
        resize_window = tk.Toplevel(root)
        resize_window.title("Resize Image")
        resize_window.geometry("300x220")
        resize_window.resizable(False, False)

        # Add Warning
        tk.Label(resize_window, text="WARNING:\nImage quality may be reduced after resizing.").pack(pady=5)
        
        # Add input fields and labels
        tk.Label(resize_window, text="Width:").pack(pady=5)
        width_entry = tk.Entry(resize_window)
        width_entry.pack(pady=5)
        
        tk.Label(resize_window, text="Height:").pack(pady=5)
        height_entry = tk.Entry(resize_window)
        height_entry.pack(pady=5)
        
        # Add OK, Cancel, and Reset buttons
        button_frame = tk.Frame(resize_window)
        button_frame.pack(pady=10)
        tk.Button(button_frame, text="OK", command=apply_resize).pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame, text="Cancel", command=cancel_resize).pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame, text="Reset Original Size", command=reset_size).pack(side=tk.LEFT, padx=5)

    def translate_image(self, x, y):
        M = np.float32([
            [1, 0, x],
            [0, 1, y]
        ])
        h = self.processed_image.shape[0]
        w = self.processed_image.shape[1]
        self.image = self.processed_image = cv2.warpAffine(self.processed_image, M, (w, h))
        self.dy += y
        self.dx += x
        self.refresh_image()

    def open_translate_window(self):
        def reset_translation():
            x = -self.dx / 2
            y = -self.dy / 2
            self.translate_image(x, y)
            self.dx = self.dy = 0
            translate_window.destroy()  # Close the resize window

        def apply_translation():
            try:
                x = int(x_entry.get())
                y = int(y_entry.get())
                self.translate_image(x, y)
                translate_window.destroy()  # Close the resize window
            except ValueError:
                messagebox.showerror("Invalid Input", "Please enter valid integers for X and Y.")

        def cancel_translation():
            translate_window.destroy()  # Close the resize window
        
        # Create the pop-up window
        translate_window = tk.Toplevel(root)
        translate_window.title("Translate Image")
        translate_window.geometry("300x220")
        translate_window.resizable(False, False)

        # Add Warning
        tk.Label(translate_window, text="WARNING:\nParts of the image may be lost after translation.").pack(pady=5)
        
        # Add input fields and labels
        tk.Label(translate_window, text="Right:").pack(pady=5)
        x_entry = tk.Entry(translate_window)
        x_entry.pack(pady=5)
        
        tk.Label(translate_window, text="Down:").pack(pady=5)
        y_entry = tk.Entry(translate_window)
        y_entry.pack(pady=5)
        
        # Add OK, Cancel, and Reset buttons
        button_frame = tk.Frame(translate_window)
        button_frame.pack(pady=10)
        tk.Button(button_frame, text="OK", command=apply_translation).pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame, text="Cancel", command=cancel_translation).pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame, text="Reset Original Position", command=reset_translation).pack(side=tk.LEFT, padx=5)
    
    def show_original(self):
        if self.image is not None:
            self.processed_image = self.image = self.original_image
            self.current_mode = None  # Reset current mode
            self.refresh_image()

    def apply_negative(self):
        # Note: Negative is the same as CMY
        if self.image is not None:
            self.processed_image = self.image = 255 - self.original_image
            self.refresh_image()

    def apply_grayscale(self):
        if self.image is not None:
            self.processed_image = self.image = cv2.cvtColor(self.original_image, cv2.COLOR_RGB2GRAY)
            self.refresh_image()

    def apply_inverse_grayscale(self):
        if self.image is not None:
            self.processed_image = self.image = 255 - cv2.cvtColor(self.original_image, cv2.COLOR_RGB2GRAY)
            self.refresh_image()

    def activate_binary(self):
        if self.image is not None:
            self.current_mode = 'binary'
            self.update_binary_image(None)

    def activate_inverse_binary(self):
        if self.image is not None:
            self.current_mode = 'inverse_binary'
            self.update_binary_image(None)
    
    def apply_hsv(self):
        if self.image is not None:
            self.processed_image = self.image = cv2.cvtColor(self.original_image, cv2.COLOR_RGB2HSV_FULL)
            self.refresh_image()
    
    def apply_lab(self):
        if self.image is not None:
            self.processed_image = self.image = cv2.cvtColor(self.original_image, cv2.COLOR_RGB2LAB)
            self.refresh_image()

    def apply_mode(self, event):
        self.threshold_slider.pack_forget()  # Hide the slider initially
        selected_mode = self.mode_combobox.get()

        if selected_mode == "Original Colors":
            self.show_original()
        elif selected_mode == "Negative (CMY)":
            self.apply_negative()
        elif selected_mode == "Grayscale":
            self.apply_grayscale()
        elif selected_mode == "Inverse Grayscale":
            self.apply_inverse_grayscale()
        elif selected_mode == "Binary":
            self.activate_binary()
            self.threshold_slider.pack(side=tk.TOP, padx=10, pady=5)
        elif selected_mode == "Inverse Binary":
            self.activate_inverse_binary()
            self.threshold_slider.pack(side=tk.TOP, padx=10, pady=5)
        elif selected_mode == "HSV":
            self.apply_hsv()
        elif selected_mode == "LAB":
            self.apply_lab()
        self.enable_effect(None)
    
    def enable_effect(self, event):
        self.quantization_combobox.pack_forget()
        self.dithering_combobox.pack_forget()
        selected_effect = self.effect_combobox.get()
        
        if selected_effect == "None":
            self.processed_image = self.image
            self.refresh_image()
        elif selected_effect == "Quantization":
            self.apply_quantization(None)
            self.quantization_combobox.pack(side=tk.LEFT, padx=5, pady=5)
        elif selected_effect == "Dithering":
            if self.mode_combobox.get() == "Grayscale" or self.mode_combobox.get() == "Inverse Grayscale":
                self.apply_dithering(None)
                self.dithering_combobox.pack(side=tk.LEFT, padx=5, pady=5)
            else:
                messagebox.showwarning("Warning", "Dithering is only applied on grayscale images\nit won\'t be applied on current color mode")
        elif selected_effect == "Median Cut":
            # Not Added Yet
            messagebox.showwarning("Work in progress", "This function is not yet ready")
            return
    
    def apply_quantization(self, event):
        q_option = self.quantization_combobox.get()

        if q_option == "32-Bit":
            n = 5
        elif q_option == "16-Bit":
            n = 4
        elif q_option == "8-Bit":
            n = 3
        elif q_option == "4-Bit":
            n = 2

        bins = np.linspace(0, self.image.max(), 2 ** n)
        digi_img = np.digitize(self.image, bins)
        digi_img = (np.vectorize(bins.tolist().__getitem__)(digi_img - 1).astype(int))
        self.processed_image = np.uint8(digi_img)
        self.refresh_image()

    def apply_dithering(self, event):
        d_option = self.dithering_combobox.get()
        dithering_matrix = []

        if d_option == "2x2":
            dithering_matrix = bayer_matrix(0)
        elif d_option == "3x3":
            dithering_matrix = [
                [0, 7, 3],
                [6, 5, 2],
                [4, 1, 8]
            ]
        elif d_option == "4x4":
            dithering_matrix = bayer_matrix(1)
        elif d_option == "8x8":
            dithering_matrix = bayer_matrix(2)
        elif d_option == "16x16":
            dithering_matrix = bayer_matrix(3)
        elif d_option == "32x32":
            dithering_matrix = bayer_matrix(4)
        elif d_option == "64x64":
            dithering_matrix = bayer_matrix(5)
        
        rows, cols = self.image.shape[0], self.image.shape[1]
        n = len(dithering_matrix)
        new_range_divider = 256 / (n * n + 1)
        norm_img = self.image // new_range_divider
        dithered_img = np.zeros_like(self.image)
        for x in range(rows):
            for y in range(cols):
                i = x % n
                j = y % n
                if norm_img[x][y] > dithering_matrix[i][j]:
                    dithered_img[x][y] = 255
                else:
                    dithered_img[x][y] = 0
        self.processed_image = dithered_img
        self.refresh_image()


    def update_binary_image(self, event):
        if self.image is None:
            return
        threshold = self.threshold_slider.get()
        gray_image = cv2.cvtColor(self.original_image, cv2.COLOR_RGB2GRAY)

        if self.current_mode == 'binary':
            self.processed_image = self.image = cv2.threshold(gray_image, threshold, 255, cv2.THRESH_BINARY)[1]
        elif self.current_mode == 'inverse_binary':
            self.processed_image = self.image = cv2.threshold(gray_image, threshold, 255, cv2.THRESH_BINARY_INV)[1]
        
        self.refresh_image()
    
    def show_gray_hist(self):
        if self.original_image.any:
        # Convert the image to grayscale and get the pixel values
            gray_img = self.original_image
            gray_img = cv2.cvtColor(gray_img, cv2.COLOR_RGB2GRAY)

            # Create a new Tkinter window for the histogram
            hist_window = tk.Toplevel(root)
            hist_window.title("Normal Grayscale Image Histogram")
            hist_window.resizable(False, False)

            gray_hist = cv2.calcHist([gray_img], [0], None, [256], [0, 255])
            # Plot the histogram using Matplotlib
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(gray_hist)
            ax.set_title("Grayscale Histogram")
            ax.set_xlabel("Pixel Intensity")
            ax.set_ylabel("Frequency")

            # Embed the plot into the Tkinter window
            canvas = FigureCanvasTkAgg(fig, master=hist_window)
            canvas.get_tk_widget().pack()
            canvas.draw()

            # Ensure the canvas is properly destroyed when the window is closed
            def on_close_histogram():
                canvas.get_tk_widget().destroy()  # Destroy the canvas widget
                plt.close(fig)  # Close the Matplotlib figure
                hist_window.destroy()  # Destroy the Tkinter window

            hist_window.protocol("WM_DELETE_WINDOW", on_close_histogram)
    
    def show_color_hist(self):
        if self.original_image.any:
        # Convert the image to grayscale and get the pixel values
            img = self.original_image
            # Create a new Tkinter window for the histogram
            hist_window = tk.Toplevel(root)
            hist_window.title("Normal RGB Image Histogram")
            hist_window.resizable(False, False)

            # Plot the histogram using Matplotlib
            fig, ax = plt.subplots(figsize=(6, 4))
            color = ['r', 'g', 'b']
            for i in range(len(color)):
                color_hist = cv2.calcHist([img], [i], None, [256], [0, 255])
                ax.plot(color_hist, color[i])
            ax.set_title("RGB Histogram")
            ax.set_xlabel("Pixel Intensity")
            ax.set_ylabel("Frequency")

            # Embed the plot into the Tkinter window
            canvas = FigureCanvasTkAgg(fig, master=hist_window)
            canvas.get_tk_widget().pack()
            canvas.draw()

            # Ensure the canvas is properly destroyed when the window is closed
            def on_close_histogram():
                canvas.get_tk_widget().destroy()  # Destroy the canvas widget
                plt.close(fig)  # Close the Matplotlib figure
                hist_window.destroy()  # Destroy the Tkinter window

            hist_window.protocol("WM_DELETE_WINDOW", on_close_histogram)
    
    def save_image(self):
        if not self.processed_image.any:
            messagebox.showerror("Error", "No processed image to save!")
            return

        filepath = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("Image files", "*.jpg;*.png;*.jpeg")])
        if not filepath:
            return

        # Determine the image type based on shape
        if len(self.processed_image.shape) == 2:  # Grayscale or binary
            cv2.imwrite(filepath, self.processed_image)
        else:  # Color image
            cv2.imwrite(filepath, cv2.cvtColor(self.processed_image, cv2.COLOR_BGR2RGB))

        messagebox.showinfo("Success", "Image saved successfully!")

# Main loop
if __name__ == "__main__":
    root = tk.Tk()
    app = ImageProcessingApp(root)
    root.mainloop()
