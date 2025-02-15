import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import threading
import warnings

import customtkinter as ctk
import cv2
import numpy as np
import torch
from PIL import Image
from sklearn.cluster import KMeans
from transformers import (DetrForObjectDetection, DetrImageProcessor,
                          VisionEncoderDecoderModel, ViTImageProcessor)

warnings.filterwarnings('ignore')

class InquiView(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("InquiView - Visual Question Generator")
        self.geometry("1400x800")
        ctk.set_appearance_mode("dark")
        
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        
        self.camera_active = False
        self.camera_thread = None
        self.current_image = None
        self.current_image_array = None
        
        self.setup_models()
        self.create_widgets()
        
    def setup_models(self):
        try:
            print("Loading models...")
            self.detector = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
            self.detection_model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
            
            self.processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
            self.vision_model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
            print("Models loaded successfully")
            
        except Exception as e:
            print(f"Error loading models: {e}")
            
    def create_widgets(self):
        # Left Frame
        self.left_frame = ctk.CTkFrame(self)
        self.left_frame.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")
        
        title_label = ctk.CTkLabel(
            self.left_frame,
            text="Image Input",
            font=ctk.CTkFont(size=20, weight="bold")
        )
        title_label.pack(pady=10)
        
        # Buttons
        button_frame = ctk.CTkFrame(self.left_frame)
        button_frame.pack(pady=10, padx=20, fill="x")
        
        self.camera_button = ctk.CTkButton(
            button_frame,
            text="📷 Start Camera",
            command=self.toggle_camera,
            height=40
        )
        self.camera_button.pack(side="left", padx=5, expand=True)
        
        self.upload_button = ctk.CTkButton(
            button_frame,
            text="📁 Upload Image",
            command=self.upload_image,
            height=40
        )
        self.upload_button.pack(side="left", padx=5, expand=True)
        
        # Image display
        self.image_frame = ctk.CTkFrame(self.left_frame)
        self.image_frame.pack(pady=10, padx=20, fill="both", expand=True)
        
        self.image_label = ctk.CTkLabel(self.image_frame, text="")
        self.image_label.pack(pady=10, expand=True)
        
        # Placeholder image
        placeholder = Image.new('RGB', (640, 480), color='#2B2B2B')
        self.current_image = ctk.CTkImage(placeholder, size=(640, 480))
        self.image_label.configure(image=self.current_image)
        
        # Generate button
        self.generate_button = ctk.CTkButton(
            self.left_frame,
            text="🔄 Analyze & Generate Questions",
            command=self.analyze_and_generate,
            height=50,
            font=ctk.CTkFont(size=16, weight="bold"),
            fg_color="#28a745",
            hover_color="#218838"
        )
        self.generate_button.pack(pady=10, padx=20, fill="x")
        
        # Style selection
        style_frame = ctk.CTkFrame(self.left_frame)
        style_frame.pack(pady=10, padx=20, fill="x")
        
        style_label = ctk.CTkLabel(style_frame, text="Analysis Style:")
        style_label.pack(side="left", padx=5)
        
        self.style_var = ctk.StringVar(value="Detailed")
        self.style_menu = ctk.CTkOptionMenu(
            style_frame,
            values=["Detailed", "Creative", "Technical", "Educational"],
            variable=self.style_var
        )
        self.style_menu.pack(side="left", padx=5)
        
        # Right Frame
        self.right_frame = ctk.CTkFrame(self)
        self.right_frame.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")
        
        questions_label = ctk.CTkLabel(
            self.right_frame,
            text="Generated Questions",
            font=ctk.CTkFont(size=20, weight="bold")
        )
        questions_label.pack(pady=10)
        
        self.questions_scroll = ctk.CTkScrollableFrame(self.right_frame)
        self.questions_scroll.pack(pady=10, padx=20, fill="both", expand=True)
        
        self.show_welcome()
        
    def show_welcome(self):
        welcome_frame = ctk.CTkFrame(self.questions_scroll, fg_color="transparent")
        welcome_frame.pack(pady=20, padx=20, fill="x")
        
        welcome_text = ("👋 Welcome to InquiView!\n\n"
                       "1. Upload an image or use your camera\n"
                       "2. Click 'Analyze & Generate Questions' to process\n"
                       "3. Questions will be based on actual image content\n\n"
                       "Ready to explore your images?")
        
        welcome_label = ctk.CTkLabel(
            welcome_frame,
            text=welcome_text,
            font=ctk.CTkFont(size=14),
            justify="left",
            wraplength=400
        )
        welcome_label.pack()
        
    def analyze_and_generate(self):
        if self.current_image_array is None:
            self.show_error("Please upload or capture an image first!")
            return
            
        try:
            # Clear previous output
            for widget in self.questions_scroll.winfo_children():
                widget.destroy()
                
            # Show analysis in progress
            self.show_status("Analyzing image...")
            
            # Process image with detection model
            inputs = self.detector(images=self.current_image_array, return_tensors="pt")
            outputs = self.detection_model(**inputs)
            
            # Get detections
            target_sizes = torch.tensor([self.current_image_array.shape[:2]])
            results = self.detector.post_process_object_detection(
                outputs, target_sizes=target_sizes, threshold=0.7
            )[0]
            
            # Analyze scene
            scene_info = self.analyze_scene(self.current_image_array, results)
            
            # Generate questions
            questions = self.generate_contextual_questions(scene_info)
            
            # Display results
            if questions:
                # Show scene analysis
                analysis_frame = ctk.CTkFrame(self.questions_scroll)
                analysis_frame.pack(pady=5, padx=10, fill="x")
                
                title = ctk.CTkLabel(
                    analysis_frame,
                    text="Scene Analysis",
                    font=ctk.CTkFont(size=16, weight="bold")
                )
                title.pack(pady=5, padx=10)
                
                for key, value in scene_info.items():
                    if value:  # Only show non-empty analyses
                        info = ctk.CTkLabel(
                            analysis_frame,
                            text=f"{key.replace('_', ' ').title()}: {str(value)}",
                            wraplength=400,
                            justify="left"
                        )
                        info.pack(pady=2, padx=10)
                
                # Show questions
                question_frame = ctk.CTkFrame(self.questions_scroll)
                question_frame.pack(pady=10, padx=10, fill="x")
                
                for i, question in enumerate(questions, 1):
                    q_label = ctk.CTkLabel(
                        question_frame,
                        text=f"Q{i}: {question}",
                        wraplength=400,
                        font=ctk.CTkFont(size=14),
                        justify="left"
                    )
                    q_label.pack(pady=5, padx=10, anchor="w")
            else:
                self.show_error("Could not generate meaningful questions for this image.")
                
        except Exception as e:
            print(f"Analysis error: {e}")
            self.show_error(f"Error during analysis: {str(e)}")
            
    def analyze_scene(self, image, detections):
        """Analyze the scene content and context"""
        scene_info = {
            'detected_objects': [],
            'main_subjects': [],
            'activities': [],
            'setting': [],
            'composition': [],
            'lighting': None,
            'mood': []
        }
        
        # Process detections
        for score, label, box in zip(detections["scores"], detections["labels"], detections["boxes"]):
            obj_name = self.detection_model.config.id2label[label.item()]
            confidence = score.item()
            
            if confidence > 0.7:
                scene_info['detected_objects'].append(f"{obj_name} ({confidence:.2f})")
                
                # Identify main subjects (high confidence detections)
                if confidence > 0.9:
                    scene_info['main_subjects'].append(obj_name)
                    
                # Analyze position and infer activities
                box_coords = box.tolist()
                y_position = box_coords[1] / image.shape[0]
                
                if obj_name == 'person':
                    if y_position < 0.3:
                        scene_info['activities'].append('elevated_position')
                    elif y_position > 0.7:
                        scene_info['activities'].append('ground_level_activity')
                        
        # Analyze colors and lighting
        avg_brightness = np.mean(image)
        if avg_brightness > 180:
            scene_info['lighting'] = 'bright'
            scene_info['mood'].append('vibrant')
        elif avg_brightness < 80:
            scene_info['lighting'] = 'dark'
            scene_info['mood'].append('moody')
        else:
            scene_info['lighting'] = 'balanced'
            
        # Analyze setting
        green_pixels = np.mean(image[:, :, 1] > np.maximum(image[:, :, 0], image[:, :, 2]))
        blue_pixels = np.mean(image[:, :, 2] > np.maximum(image[:, :, 0], image[:, :, 1]))
        
        if green_pixels > 0.3:
            scene_info['setting'].append('nature')
        if blue_pixels > 0.3:
            scene_info['setting'].append('water/sky prominent')
            
        # Analyze composition
        edges = cv2.Canny(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), 100, 200)
        if np.mean(edges) > 50:
            scene_info['composition'].append('strong_lines')
            
        return scene_info
        
    def generate_contextual_questions(self, scene_info):
        """Generate questions based on scene analysis"""
        questions = set()  # Use set to avoid duplicates
        style = self.style_var.get().lower()
        
        # Generate based on detected objects and activities
        for subject in scene_info['main_subjects']:
            if 'elevated_position' in scene_info['activities']:
                questions.add(f"What perspective does this elevated position provide of the {subject}?")
                questions.add(f"How does height influence the {subject}'s interaction with the environment?")
            
            if 'ground_level_activity' in scene_info['activities']:
                questions.add(f"What activity is the {subject} engaged in at ground level?")
        
        # Generate based on setting
        if 'nature' in scene_info['setting']:
            questions.add("How does the natural environment influence the scene's meaning?")
        if 'water/sky prominent' in scene_info['setting']:
            questions.add("What role does the water/sky play in the image's composition?")
            
        # Generate based on lighting and mood
        if scene_info['lighting'] == 'bright':
            questions.add("How does the bright lighting affect the emotional impact?")
        elif scene_info['lighting'] == 'dark':
            questions.add("What atmosphere does the low-key lighting create?")
            
        # Generate based on composition
        if 'strong_lines' in scene_info['composition']:
            questions.add("How do the strong compositional lines guide the viewer's eye?")
            
        # Add style-specific questions
        if style == 'detailed':
            questions.add("What subtle details reveal the story behind this moment?")
        elif style == 'creative':
            questions.add("What emotions or memories might this scene evoke?")
        elif style == 'technical':
            questions.add("How do the technical elements enhance the visual narrative?")
            
        return list(questions)

    def show_status(self, message):
        status_frame = ctk.CTkFrame(self.questions_scroll)
        status_frame.pack(pady=10, padx=20, fill="x")
        
        status_label = ctk.CTkLabel(
            status_frame,
            text=f"📊 {message}",
            font=ctk.CTkFont(size=14)
        )
        status_label.pack(pady=10)
        
    def show_error(self, message):
        error_frame = ctk.CTkFrame(self.questions_scroll, fg_color="#612020")
        error_frame.pack(pady=10, padx=20, fill="x")
        
        error_label = ctk.CTkLabel(
            error_frame,
            text=f"⚠️ Error: {message}",
            text_color="#ff7070",
            wraplength=400
        )
        error_label.pack(pady=10)
        
    def toggle_camera(self):
        if not self.camera_active:
            self.camera_active = True
            self.camera_button.configure(text="⏹️ Stop Camera")
            self.camera_thread = threading.Thread(target=self.camera_feed)
            self.camera_thread.daemon = True
            self.camera_thread.start()
        else:
            self.camera_active = False
            self.camera_button.configure(text="📷 Start Camera")
            
    def camera_feed(self):
        cap = cv2.VideoCapture(0)
        while self.camera_active:
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.current_image_array = frame_rgb
                pil_image = Image.fromarray(frame_rgb)
                pil_image.thumbnail((640, 480), Image.Resampling.LANCZOS)
                self.current_image = ctk.CTkImage(pil_image, size=pil_image.size)
                self.after(10, self.update_image)
        cap.release()
        
    def update_image(self):
        if self.current_image:
            self.image_label.configure(image=self.current_image)
        
    def upload_image(self):
        file_path = ctk.filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif")]
        )
        if file_path:
            try:
                image = cv2.imread(file_path)
                if image is not None:
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    self.current_image_array = image_rgb
                    pil_image = Image.fromarray(image_rgb)
                    pil_image.thumbnail((640, 480), Image.Resampling.LANCZOS)
                    self.current_image = ctk.CTkImage(pil_image, size=pil_image.size)
                    self.update_image()
                else:
                    self.show_error("Failed to load image!")
            except Exception as e:
                self.show_error(f"Error loading image: {str(e)}")

if __name__ == "__main__":
    app = InquiView()
    app.mainloop()