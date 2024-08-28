import cv2
import numpy as np

def capture_and_process_image(filename):
    # Step 1: Capture Thumb Impression
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)  # Width
    cap.set(4, 480)  # Height

    while True:
        ret, frame = cap.read()
        cv2.imshow('Capture Thumb Impression', frame)

        # Wait for user to press 'c' to capture the image
        if cv2.waitKey(1) & 0xFF == ord('c'):
            cv2.imwrite(filename, frame)
            print(f'Thumb Impression captured and saved as {filename}')
            break

    cap.release()
    cv2.destroyAllWindows()

    # Step 2: Process the Captured Image for Hand Detection
    image = cv2.imread(filename)
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define the range for skin color in HSV
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)

    # Threshold the HSV image to get only skin colors
    mask = cv2.inRange(hsv_image, lower_skin, upper_skin)

    # Apply morphological transformations to filter out the background noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    mask = cv2.GaussianBlur(mask, (3, 3), 0)

    # Step 3: Find Contours of the Hand
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Find the largest contour which should be the hand
    if contours:
        max_contour = max(contours, key=cv2.contourArea)
        
        # Draw the contour on the original image
        cv2.drawContours(image, [max_contour], -1, (0, 255, 0), 2)
        
        # Step 4: Convex Hull around the hand
        hull = cv2.convexHull(max_contour)
        
        # Draw the convex hull on the image
        cv2.drawContours(image, [hull], -1, (255, 0, 0), 2)

        # Step 5: Dot Representation
        for i in range(0, len(hull), 5):  # Adjust the step for more or fewer dots
            (x, y) = hull[i][0]
            cv2.circle(image, (x, y), 5, (0, 0, 255), -1)

    # Save the final processed image
    processed_filename = f'processed_{filename}'
    cv2.imwrite(processed_filename, image)

    # Display the final image with the selected hand and thumb part
    cv2.imshow('Hand and Thumb Detection', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return processed_filename

def process_image_for_matching(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Histogram equalization to improve contrast
    equalized_image = cv2.equalizeHist(image)
    
    # Apply Gaussian Blur to reduce noise
    blurred_image = cv2.GaussianBlur(equalized_image, (5, 5), 0)
    
    # Apply Morphological operations to remove small noise
    kernel = np.ones((5, 5), np.uint8)
    morph_image = cv2.morphologyEx(blurred_image, cv2.MORPH_CLOSE, kernel)
    
    # Edge detection using Canny
    edges = cv2.Canny(morph_image, 50, 150)
    
    return edges

def match_impressions(image1_path, image2_path):
    image1 = process_image_for_matching(image1_path)
    image2 = process_image_for_matching(image2_path)

    # Use SIFT for better feature detection
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(image1, None)
    kp2, des2 = sift.detectAndCompute(image2, None)

    # Use FLANN-based matcher for faster and more accurate matches
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)  # Or pass empty dictionary
    
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    # Apply ratio test to filter good matches
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    # Apply RANSAC to find the homography and filter outliers
    if len(good_matches) > 10:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()
    else:
        matchesMask = None

    draw_params = dict(matchColor=(0, 255, 0),  # Draw matches in green color
                       singlePointColor=None,
                       matchesMask=matchesMask,  # Draw only inliers
                       flags=2)

    match_img = cv2.drawMatches(image1, kp1, image2, kp2, good_matches, None, **draw_params)

    # Show the match result
    cv2.imshow('Matching Result', match_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Determine if there's a good match based on the number of good matches
    match_threshold = 15  # Increase the threshold for more accuracy
    if len(good_matches) > match_threshold:
        return True  # Match found
    else:
        return False  # No match

# Capture and process the first thumb impression
first_image_path = capture_and_process_image('first_thumb_impression.jpg')

# Capture and process the second thumb impression for matching
second_image_path = capture_and_process_image('second_thumb_impression.jpg')

# Match the thumb impressions
is_match = match_impressions(first_image_path, second_image_path)

if is_match:
    print("Thumb impressions match!")
else:
    print("No match found.")
