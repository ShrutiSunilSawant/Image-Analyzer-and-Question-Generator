# Image-Analyzer-and-Question-Generator

Image Analyzer and Question Generator is a Python-based GUI application that leverages advanced AI models to perform intelligent scene analysis on images. By combining state-of-the-art computer vision models (DETR and ViT), the application generates contextually relevant and dynamic questions based on visual content, enhancing understanding and insights from images.

Features:

Image Input Options:
Upload images from your local machine.
Capture images using your webcam.
Intelligent Scene Analysis:
Utilizes DETR for object detection.
Employs ViT for scene understanding and image captioning.
Processes aspects like lighting, composition, and contextual elements.
Dynamic Question Generation:
Automatically generates insightful, context-based questions about the visual content.
Supports multiple analysis styles (Detailed, Creative, Technical, Educational).
Responsive GUI:
Real-time camera integration with multi-threading for smooth performance.
Modern and intuitive interface using CustomTkinter.

Tech Stack
Programming Language: Python

Libraries & Frameworks:
CustomTkinter (GUI)
OpenCV (Image processing & camera integration)
NumPy (Numerical operations)
PyTorch (Deep learning)
Transformers (Pre-trained models such as DETR and ViT)
Pillow (Image manipulation)
Scikit-learn (Additional data analysis)

Other Tools:
Python threading for concurrent operations

Set Up a Virtual Environment (Optional but Recommended):
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

Install the Dependencies:
pip install customtkinter opencv-python numpy torch pillow scikit-learn transformers
Note: For PyTorch, please refer to the official installation guide to ensure compatibility with your hardware.

Usage
Run the Application:
python inquiview.py

Operating the App:

Upload Image: Click the "📁 Upload Image" button to choose an image from your computer.
Start Camera: Click the "📷 Start Camera" button to activate your webcam.
Analyze & Generate Questions: Once an image is captured or uploaded, click the "🔄 Analyze & Generate Questions" button to process the image and view the generated questions.
Known Issues
Camera Compatibility: Ensure your camera is properly connected and accessible.
Performance: The speed of analysis might vary depending on your system's GPU capabilities and processing power.

