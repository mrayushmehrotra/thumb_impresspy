# Thumb Impression Capture and Matching - Contactless Solution

This **BETA** version provides a **contactless thumb impression capture and matching system**—a solution particularly relevant in the post-COVID world, minimizing the need for physical contact. The application uses advanced computer vision techniques, making it safer and hygienic.

## Features

1. **Contactless Thumb Impression Capture:**
   - Capture an image of a thumb impression using a webcam.
   - Press **'c'** to capture images.
   
2. **Image Processing:**
   - Automatically processes the captured image to detect and isolate the thumb impression using computer vision algorithms.

3. **Thumb Impression Matching:**
   - Compares two thumb impressions to determine if they match using **SIFT (Scale-Invariant Feature Transform)**.

## Prerequisites

Ensure you have the following installed:

- **Python 3.x**
- **OpenCV**: For image capture and processing.
- **NumPy**: For numerical operations.

## Installation and Usage

Follow these steps to set up and run the application:

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/mrayushmehrotra/thumb_impresspy.git
   ```

2. **Install Dependencies:**

   ```bash
   pip3 install -r requirements.txt
   ```

3. **Run the Application:**

   ```bash
   cd thumb_impresspy
   python3 index.py
   ```

4. **Capture Thumb Impressions:**
   - Launch the application and use the webcam to capture thumb impressions.
   - Press **'c'** to capture the image.

## Future Scope

- Integration with mobile devices for wider accessibility.
- Advanced matching algorithms for enhanced accuracy.
- User-friendly UI for seamless experience.

---

This system ensures a safe, efficient, and hygienic approach to thumb impression verification, addressing the challenges of a contact-driven world.
