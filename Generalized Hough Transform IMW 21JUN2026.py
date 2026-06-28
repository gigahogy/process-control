import cv2
import numpy as np
import glob
from PIL import Image
import pytesseract
from pdf2image import convert_from_path

import tempfile

generate_images_from_pdf = False
detect_text = False

if generate_images_from_pdf:
    with tempfile.TemporaryDirectory() as path:
        images_from_path = convert_from_path('Sheets_selected_8 - P&IDs_2026-04-29_04-51-12pm.pdf', output_folder="PandID images", dpi = 200, output_file='page', fmt="jpeg")
"""
images = convert_from_path('3pagepdf.pdf')

for i in range(len(images)):
  
    images[i].save(r'PandID images/page'+ str(i) +'.jpg', 'JPEG')

print(len(images))

print(np.array(images[1]).shape)
cv2.imshow("PandID", np.array(images[1]))
cv2.waitKey(0)
cv2.destroyAllWindows()
"""
path_pattern = "Hough PandID"
PandID_paths = glob.glob(path_pattern + r"/**/*", recursive=True)
#print(PandID_paths)

path_pattern = "Hough key"
key_paths = glob.glob(path_pattern + r"/**/*", recursive=True)
#print(key_paths)

# Load images
#template = cv2.imread("valve image.jpg", cv2.IMREAD_GRAYSCALE)

#Comment out if don't want to rotate
#template = cv2.rotate(template, cv2.ROTATE_90_CLOCKWISE)

def box_detection(template, image, img_edges, vote_fraction, threshold_value):
    #image = cv2.imread("PandID with valves.jpg")
    template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    

    # Edge maps
    #templ_edges = cv2.Canny(template, 50, 150)
    #img_edges = cv2.Canny(gray, 50, 150)


    #ret, templ_edges = cv2.threshold(template, 127, 255, cv2.THRESH_BINARY)
    #ret, img_edges = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    #30 or 230, try both or other values and maybe thresholding, affects line thickness in template
    ret, templ_edges = cv2.threshold(template, threshold_value, 255, cv2.THRESH_BINARY_INV)
    #print(cv2.countNonZero(templ_edges))
    #print(templ_edges.shape)
    max_possible_votes = cv2.countNonZero(templ_edges)
    vote_count = int(max_possible_votes * vote_fraction)

    #This shows what the template looks like and is pretty important to diagnostics/ explaining process.  Uncomment line below to show when teaching.
    #cv2.imshow("templ_edges", templ_edges)



    positions = []
    votes = []
    height = []
    width = []
    for i in range(4):
        # Create Ballard detector
        gh = cv2.createGeneralizedHoughBallard()

         # Optional tuning
        gh.setVotesThreshold(vote_count)
        
        # Set template
        gh.setTemplate(templ_edges)
        #h, w = template.shape[:2]
        h, w = templ_edges.shape
        #print(template.shape[:2])
        #print(templ_edges.shape)
        #print(templ_edges.shape)
        #print(i)
        #print(h)
        #print(w)
        templ_edges = cv2.rotate(templ_edges, cv2.ROTATE_90_CLOCKWISE)
        
        #gh.setMinDist(min(template.shape[:2]))

        # Detect
        positions_rotation, votes_rotation = gh.detect(img_edges)
        #print(type(positions_rotation))
        #print(type(votes_rotation))
        if positions_rotation is not None:
            #print(positions)
            height += [h]*positions_rotation.shape[1]
            width += [w]*positions_rotation.shape[1]
            positions += [positions_rotation]
        if votes_rotation is not None:
            votes += [votes_rotation]

    if positions != []:
        #print(positions)
        #print(votes)
        positions = np.concatenate(positions, axis = 1)
        votes = np.concatenate(votes, axis = 1)
        #print(positions)
        #print(votes)

        #print(height)
        #print(width)

        #print(len(positions[0]))
        #print(len(height))
        #h, w = template.shape[:2]

        # OpenCV often returns shape (1, N, 4)
        detections = positions.reshape(-1, positions.shape[-1])
        #print(detections.shape[0])
        #print(detections)

        #print(type([int(vote) for vote in votes[0,:,0]]))
        boxes = [[int(round(det[0])), int(round(det[1])), width[i], height[i]] for i, det in enumerate(detections)]
        scores = [int(vote) for vote in votes[0,:,0]]
        score_threshold = 0
        nms_threshold = 0.3
        
        indices = cv2.dnn.NMSBoxes(
        boxes,
        scores,
        score_threshold,
        nms_threshold
        )
        #print(indices)

        #readjus indices to apply non maximum suppression
        detections = [detections[index] for index in indices]
        width = [width[index] for index in indices]
        height = [height[index] for index in indices]

        for i, det in enumerate(detections):
            x = int(round(det[0]))
            y = int(round(det[1]))

            # Bounding box using template size
            x1 = x - width[i] // 2
            y1 = y - height[i] // 2
            x2 = x + width[i] // 2
            y2 = y + height[i] // 2

            cv2.rectangle(
                image,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            # Center point
            cv2.circle(
                image,
                (x, y),
                4,
                (0, 0, 255),
                -1
            )

    return img_edges

#current images use 100% zoom and 0.22 vote_fraction
vote_fraction = 0.25 #fraction of overlapping pixels required for a positive result

threshold_value = 30
"""
image_path = "PandID with valves.jpg"
template_path = "valve image.jpg"

template = cv2.imread(template_path)
image = cv2.imread(image_path)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
ret, img_edges = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY_INV) #ret is a place holder return that isn't used anywhere else

img_edges = box_detection(template, image, img_edges, vote_count)

template = cv2.rotate(template, cv2.ROTATE_90_CLOCKWISE)
img_edges = box_detection(template, image, img_edges, vote_count)
"""

for i, PandID_path in enumerate(PandID_paths):
    #print(pytesseract.image_to_string(Image.open(PandID_path)))
    #pytesseract.image_to_string(Image.fromarray(img_rgb))
    PandID_image = cv2.imread(PandID_path)
    if detect_text:
        print(pytesseract.image_to_string(Image.fromarray(PandID_image)))
        print("rotated")
        PandID_image_rotated = cv2.rotate(PandID_image, cv2.ROTATE_90_CLOCKWISE)
        cv2.imshow("rotated", PandID_image_rotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        print(pytesseract.image_to_string(Image.fromarray(PandID_image_rotated)))

    PandID_gray = cv2.cvtColor(PandID_image, cv2.COLOR_BGR2GRAY)
    ret, PandID_edges = cv2.threshold(PandID_gray, threshold_value, 255, cv2.THRESH_BINARY_INV) #ret is a place holder return that isn't used anywhere else
    for key_path in key_paths:
        key_image = cv2.imread(key_path)
        PandID_edges = box_detection(key_image, PandID_image, PandID_edges, vote_fraction, threshold_value)

        #key_image_90_rotated = cv2.rotate(key_image, cv2.ROTATE_90_CLOCKWISE)
        #PandID_edges = box_detection(key_image_90_rotated, PandID_image, PandID_edges, vote_fraction, threshold_value)

    img_edges = PandID_edges
    image = PandID_image

    #cv2.imshow("Ballard Detection", image)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    # Probabilistic Hough Transform (returns line segment endpoints)
    lines = cv2.HoughLinesP(
        img_edges,
        rho=1,
        theta=np.pi / 180,
        threshold=80,
        minLineLength=30,
        maxLineGap=1
    )

    # Draw detected line segments robustly
    if lines is not None:
        for x1, y1, x2, y2 in lines[:, 0]:
            cv2.line(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
    else:
        print("No lines detected.")

    # Show the result
    #cv2.imshow("Detected Lines", image)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    cv2.imwrite(f"PandID results//test_{i}.jpg", image)

############################
#text detection using neural networks
#https://tesseractocr.org/
#https://pypi.org/project/pytesseract/

#for PandID_path in PandID_paths:
    #print(pytesseract.image_to_string(Image.open(PandID_path)))

#https://www.geeksforgeeks.org/python/convert-pdf-to-image-using-python/
#https://github.com/oschwartz10612/poppler-windows/releases/

    

"""
import cv2
import numpy as np

# Load template and target image
template = cv2.imread("valve image.jpg", cv2.IMREAD_GRAYSCALE)
image = cv2.imread("PandID with valves.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Edge detection
templ_edges = cv2.Canny(template, 50, 150)
img_edges = cv2.Canny(gray, 50, 150)

# Create Generalized Hough detector
gh = cv2.createGeneralizedHoughGuil()

# Optional tuning
angle_votes = 30
scale_votes = 30
position_votes = 30
#gh.setAngleThresh(angle_votes)
#gh.setScaleThresh(scale_votes)
gh.setPosThresh(position_votes)

# Set template
gh.setTemplate(templ_edges)

gh.setMinScale(1)
gh.setMaxScale(1.1)
gh.setScaleStep(0.5)

#Search range
gh.setMinAngle(0)
gh.setMaxAngle(90)

# Rotation increment (degrees)
gh.setAngleStep(90)


# Detect
positions, votes = gh.detect(img_edges)

if positions is not None:

    h, w = template.shape[:2]

    for p in positions[0]:
        # OpenCV returns:
        # x, y, scale, angle
        x, y, scale, angle = p[:4]

        # Template corners
        corners = np.array([
            [0, 0],
            [w, 0],
            [w, h],
            [0, h]
        ], dtype=np.float32)

        # Template center
        center = np.array([w / 2, h / 2])

        # Rotation matrix
        theta = np.deg2rad(angle)
        R = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta),  np.cos(theta)]
        ])

        # Transform corners
        transformed = []
        for c in corners:
            pt = (c - center) * scale
            pt = R @ pt
            pt += np.array([x, y])
            transformed.append(pt)

        transformed = np.array(transformed, dtype=np.int32)

        # Draw rotated bounding box
        cv2.polylines(
            image,
            [transformed],
            True,
            (0, 255, 0),
            2
        )

        # Axis-aligned bounding box
        bx, by, bw, bh = cv2.boundingRect(transformed)

        cv2.rectangle(
            image,
            (bx, by),
            (bx + bw, by + bh),
            (0, 0, 255),
            2
        )

cv2.imshow("Result", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""
