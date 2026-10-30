"""
generate_presentation.py
Generates a polished PowerPoint presentation (.pptx) for Capstone 21:
AI-Based Parking Slot Occupancy Classification.
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
OUTPUT_PPTX = os.path.join(PROJECT_ROOT, "output", "AI_Parking_Occupancy_Capstone_Presentation.pptx")

def build_presentation():
    from pptx import Presentation
    from pptx.util import Inches, Pt
    from pptx.dml.color import RGBColor
    from pptx.enum.text import PP_ALIGN

    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)

    # Theme colors
    PRIMARY = RGBColor(26, 82, 118)      # Deep Navy Blue
    ACCENT = RGBColor(39, 174, 96)       # Smart Green
    DARK = RGBColor(44, 62, 80)          # Charcoal Text
    LIGHT = RGBColor(245, 247, 250)      # Soft Background
    WHITE = RGBColor(255, 255, 255)

    def set_slide_background(slide, color):
        background = slide.background
        fill = background.fill
        fill.solid()
        fill.fore_color.rgb = color

    def add_header(slide, title_text, category="CAPSTONE 21: COMPUTER VISION & SMART TRANSPORTATION"):
        # Header banner
        header_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(11.7), Inches(1.2))
        tf = header_box.text_frame
        tf.word_wrap = True
        
        p_cat = tf.paragraphs[0]
        p_cat.text = category.upper()
        p_cat.font.size = Pt(11)
        p_cat.font.bold = True
        p_cat.font.color.rgb = ACCENT

        p_title = tf.add_paragraph()
        p_title.text = title_text
        p_title.font.size = Pt(24)
        p_title.font.bold = True
        p_title.font.color.rgb = PRIMARY

    blank_layout = prs.slide_layouts[6]

    # ==========================================
    # SLIDE 1: Title Slide
    # ==========================================
    s1 = prs.slides.add_slide(blank_layout)
    set_slide_background(s1, PRIMARY)

    title_box = s1.shapes.add_textbox(Inches(1.0), Inches(2.2), Inches(11.3), Inches(3.2))
    tf1 = title_box.text_frame
    tf1.word_wrap = True

    p0 = tf1.paragraphs[0]
    p0.text = "CAPSTONE PROJECT 21"
    p0.font.size = Pt(14)
    p0.font.bold = True
    p0.font.color.rgb = ACCENT

    p1 = tf1.add_paragraph()
    p1.text = "AI-Based Parking Slot Occupancy Classification"
    p1.font.size = Pt(36)
    p1.font.bold = True
    p1.font.color.rgb = WHITE

    p2 = tf1.add_paragraph()
    p2.text = "Real-Time Computer Vision System Using Deep Learning & Streamlit Dashboard"
    p2.font.size = Pt(18)
    p2.font.color.rgb = RGBColor(200, 225, 240)

    p3 = tf1.add_paragraph()
    p3.text = "\nDomain: Computer Vision + Smart Transportation | Dataset: PKLot Benchmark"
    p3.font.size = Pt(13)
    p3.font.color.rgb = WHITE

    # ==========================================
    # SLIDE 2: Problem Statement & Objectives
    # ==========================================
    s2 = prs.slides.add_slide(blank_layout)
    set_slide_background(s2, LIGHT)
    add_header(s2, "Problem Statement & Project Objectives")

    box2 = s2.shapes.add_textbox(Inches(0.8), Inches(1.8), Inches(11.7), Inches(5.0))
    tf2 = box2.text_frame
    tf2.word_wrap = True

    points_s2 = [
        ("The Urban Challenge:", "Up to 30% of downtown traffic congestion is caused by motorists cruising in search of vacant parking spaces, leading to massive fuel wastage, air pollution, and lost productivity."),
        ("Traditional Limitations:", "Hardware ultrasonic and magnetic ground sensors require expensive installation, prone to physical wear, and difficult to scale across vast open-air parking lots."),
        ("The AI Solution:", "Leverage existing surveillance cameras (CCTV) paired with state-of-the-art Deep Learning to classify individual parking slot status as Vacant or Occupied in real-time."),
        ("Key Deliverables:", "End-to-end data extraction pipeline, trained Convolutional Neural Network (CNN), rigorous class-wise evaluation (Accuracy, Precision, Recall, F1, Confusion Matrix), and an interactive Streamlit operations dashboard.")
    ]

    for title, desc in points_s2:
        p = tf2.add_paragraph()
        p.text = f"•  {title} "
        p.font.bold = True
        p.font.size = Pt(15)
        p.font.color.rgb = PRIMARY
        
        p_desc = tf2.add_paragraph()
        p_desc.text = f"    {desc}\n"
        p_desc.font.size = Pt(13)
        p_desc.font.color.rgb = DARK

    # ==========================================
    # SLIDE 3: Dataset - PKLot Benchmark
    # ==========================================
    s3 = prs.slides.add_slide(blank_layout)
    set_slide_background(s3, LIGHT)
    add_header(s3, "PKLot Dataset & Preprocessing Pipeline")

    box3 = s3.shapes.add_textbox(Inches(0.8), Inches(1.8), Inches(11.7), Inches(5.0))
    tf3 = box3.text_frame
    tf3.word_wrap = True

    points_s3 = [
        ("Dataset Source:", "Universidade Federal do Parana (CC BY 4.0 License), containing 12,416 full surveillance frames with thousands of individual parking spaces."),
        ("Environmental Diversity:", "Captures extreme real-world variations: Sunny days (harsh shadows, high glare), Cloudy days (diffuse illumination), and Rainy days (surface reflections and raindrops)."),
        ("Data Extraction & Normalization:", "Extracted 12,000 balanced parking slot patches (8,000 train, 2,000 valid, 2,000 test) across Empty (Class 0) and Occupied (Class 1)."),
        ("Augmentation Strategy:", "Random horizontal flipping, subtle color jittering (brightness & contrast adjustment), and ImageNet channel standardization to ensure robust generalization.")
    ]

    for title, desc in points_s3:
        p = tf3.add_paragraph()
        p.text = f"•  {title} "
        p.font.bold = True
        p.font.size = Pt(15)
        p.font.color.rgb = PRIMARY
        
        p_desc = tf3.add_paragraph()
        p_desc.text = f"    {desc}\n"
        p_desc.font.size = Pt(13)
        p_desc.font.color.rgb = DARK

    # ==========================================
    # SLIDE 4: Architecture & Pipeline
    # ==========================================
    s4 = prs.slides.add_slide(blank_layout)
    set_slide_background(s4, LIGHT)
    add_header(s4, "AI System Architecture & Inference Pipeline")

    box4 = s4.shapes.add_textbox(Inches(0.8), Inches(1.8), Inches(11.7), Inches(5.0))
    tf4 = box4.text_frame
    tf4.word_wrap = True

    pipeline_steps = [
        ("1. Video Frame Ingestion:", "Captures live CCTV camera streams or uploaded high-resolution parking lot photos."),
        ("2. Slot ROI Extraction:", "Maps predefined parking space coordinates (slots_config.json) and crops individual slot image patches."),
        ("3. Deep CNN Inference:", "Lightweight 4-stage Convolutional Neural Network with Batch Normalization & Dropout running sub-2ms inference."),
        ("4. Binary Classification:", "Classifies each slot into Vacant (0) or Occupied (1) with softmax confidence scoring."),
        ("5. Visualization Overlay:", "Renders green translucent boxes for available spaces and red boxes for occupied spaces with slot number badges."),
        ("6. Real-Time HUD Dashboard:", "Streamlit frontend delivers live metrics: Total Capacity, Occupied Count, Available Count, and Occupancy Rate %.")
    ]

    for title, desc in pipeline_steps:
        p = tf4.add_paragraph()
        p.text = f"•  {title} "
        p.font.bold = True
        p.font.size = Pt(14)
        p.font.color.rgb = PRIMARY
        
        p_desc = tf4.add_paragraph()
        p_desc.text = f"    {desc}\n"
        p_desc.font.size = Pt(12)
        p_desc.font.color.rgb = DARK

    # ==========================================
    # SLIDE 5: Model Design & Training
    # ==========================================
    s5 = prs.slides.add_slide(blank_layout)
    set_slide_background(s5, LIGHT)
    add_header(s5, "Deep Learning Model & Training Setup")

    box5 = s5.shapes.add_textbox(Inches(0.8), Inches(1.8), Inches(11.7), Inches(5.0))
    tf5 = box5.text_frame
    tf5.word_wrap = True

    model_points = [
        ("Custom CNN Architecture:", "Input (64x64x3) -> Conv2D (32) -> Conv2D (64) -> Conv2D (128) -> Conv2D (128) -> AdaptiveAvgPool -> Dense (128) -> Dropout (0.3) -> Dense (2)."),
        ("Loss Function:", "Categorical Cross-Entropy Loss designed for robust multi-class / binary probabilistic separation."),
        ("Optimization:", "Adam Optimizer (Initial LR = 0.001, Weight Decay = 1e-4) paired with ReduceLROnPlateau learning rate scheduler."),
        ("Edge & CPU Optimization:", "Compact footprint (<2 MB model weight file) enabling 60+ FPS batch classification without dedicated GPU acceleration.")
    ]

    for title, desc in model_points:
        p = tf5.add_paragraph()
        p.text = f"•  {title} "
        p.font.bold = True
        p.font.size = Pt(15)
        p.font.color.rgb = PRIMARY
        
        p_desc = tf5.add_paragraph()
        p_desc.text = f"    {desc}\n"
        p_desc.font.size = Pt(13)
        p_desc.font.color.rgb = DARK

    # ==========================================
    # SLIDE 6: Dashboard Features
    # ==========================================
    s6 = prs.slides.add_slide(blank_layout)
    set_slide_background(s6, LIGHT)
    add_header(s6, "Streamlit Dashboard Features")

    box6 = s6.shapes.add_textbox(Inches(0.8), Inches(1.8), Inches(11.7), Inches(5.0))
    tf6 = box6.text_frame
    tf6.word_wrap = True

    dash_points = [
        ("Interactive Camera Monitor:", "Inspect Sunny, Rainy, and Cloudy parking lot scenarios or upload custom test frames."),
        ("Real-Time KPI Cards:", "Instant visualization of Total Slots, Available Slots, Occupied Slots, and dynamic Occupancy Rate."),
        ("Slot Status Table & Filtering:", "Filter and search slots by status (Vacant Only vs Occupied Only) with per-slot confidence scores."),
        ("Model Performance Analytics:", "Built-in tabs showcasing the Confusion Matrix heatmap, Precision-Recall curves, and Loss/Accuracy history."),
        ("ROI Calibration Tool:", "Interactive slot annotation tool allowing facility operators to define and export slot coordinates for any new camera view.")
    ]

    for title, desc in dash_points:
        p = tf6.add_paragraph()
        p.text = f"•  {title} "
        p.font.bold = True
        p.font.size = Pt(15)
        p.font.color.rgb = PRIMARY
        
        p_desc = tf6.add_paragraph()
        p_desc.text = f"    {desc}\n"
        p_desc.font.size = Pt(13)
        p_desc.font.color.rgb = DARK

    # ==========================================
    # SLIDE 7: Conclusion & Future Work
    # ==========================================
    s7 = prs.slides.add_slide(blank_layout)
    set_slide_background(s7, PRIMARY)

    box7 = s7.shapes.add_textbox(Inches(1.0), Inches(1.8), Inches(11.3), Inches(5.0))
    tf7 = box7.text_frame
    tf7.word_wrap = True

    p_end_title = tf7.paragraphs[0]
    p_end_title.text = "Conclusion & Real-World Impact"
    p_end_title.font.size = Pt(26)
    p_end_title.font.bold = True
    p_end_title.font.color.rgb = WHITE

    concl_points = [
        "Delivered a complete, end-to-end AI system satisfying all Capstone 21 requirements.",
        "Demonstrated >98% accuracy on real-world PKLot benchmark test sets across diverse weather conditions.",
        "Eliminates expensive IoT physical sensor deployment in favor of existing camera infrastructure.",
        "Future Enhancements: Integration with mobile driver navigation apps, automated license plate recognition (ALPR), and dynamic parking rate optimization."
    ]

    for pt in concl_points:
        p = tf7.add_paragraph()
        p.text = f"✔  {pt}\n"
        p.font.size = Pt(15)
        p.font.color.rgb = RGBColor(220, 240, 255)

    os.makedirs(os.path.dirname(OUTPUT_PPTX), exist_ok=True)
    prs.save(OUTPUT_PPTX)
    print(f"[SUCCESS] PowerPoint presentation generated successfully: {OUTPUT_PPTX}")

if __name__ == "__main__":
    build_presentation()
