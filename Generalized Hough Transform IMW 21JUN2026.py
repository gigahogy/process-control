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
        images_from_path = convert_from_path(r'C:\Users\admin\Documents\Python\Sheets_selected_8 - P&IDs_2026-04-29_04-51-12pm.pdf', output_folder=r"C:\Users\admin\Documents\Python\PandID images", dpi = 200, output_file='page', fmt="jpeg")

path_pattern = r"C:\Users\admin\Documents\Python\Hough PandID"
PandID_paths = glob.glob(path_pattern + r"/**/*", recursive=True)

path_pattern = r"C:\Users\admin\Documents\Python\Hough key"
key_paths = glob.glob(path_pattern + r"/**/*", recursive=True)


#Comment out if don't want to rotate
#template = cv2.rotate(template, cv2.ROTATE_90_CLOCKWISE)

def box_detection(template_paths, image, img_edges, vote_fraction, threshold_value):

    positions = []
    votes = []
    height = []
    width = []
    coords = []
    max_votes = []
    for template_path in template_paths:
        template = cv2.imread(template_path)
        template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        

        # Edge maps
        #templ_edges = cv2.Canny(template, 50, 150)
        #img_edges = cv2.Canny(gray, 50, 150)

        #30 or 230, try both or other values and maybe thresholding, affects line thickness in template
        ret, templ_edges = cv2.threshold(template, threshold_value, 255, cv2.THRESH_BINARY_INV)
        
        max_possible_votes = cv2.countNonZero(templ_edges)
        vote_count = int(max_possible_votes * vote_fraction)

        #This shows what the template looks like and is pretty important to diagnostics/ explaining process.  Uncomment line below to show when teaching.
        #cv2.imshow("templ_edges", templ_edges)

        for i in range(4):
            templ_edges = cv2.rotate(templ_edges, cv2.ROTATE_90_CLOCKWISE)
            # Create Ballard detector
            gh = cv2.createGeneralizedHoughBallard()

             # Optional tuning
            gh.setVotesThreshold(vote_count)
            
            # Set template
            gh.setTemplate(templ_edges)
            
            h, w = templ_edges.shape
            

            coord = np.where(templ_edges)
            
            #gh.setMinDist(min(template.shape[:2]))

            # Detect
            positions_rotation, votes_rotation = gh.detect(img_edges)
            
            if positions_rotation is not None:
                
                height += [h]*positions_rotation.shape[1]
                width += [w]*positions_rotation.shape[1]
                positions += [positions_rotation]
                
                coords += [coord]*positions_rotation.shape[1]
                
                votes += [votes_rotation]
                max_votes += [max_possible_votes]*positions_rotation.shape[1]

    if positions != []:

        positions = np.concatenate(positions, axis = 1)
        
        votes = np.concatenate(votes, axis = 1)

        # OpenCV often returns shape (1, N, 4)
        detections = positions.reshape(-1, positions.shape[-1])

        boxes = [[int(round(det[0])), int(round(det[1])), width[i], height[i]] for i, det in enumerate(detections)]

        #max_possible_votes = cv2.countNonZero(templ_edges)
        #max_votes
        #scores = [vote/max_votes[index] for index, vote in enumerate(votes[0,:,0])]

        #this is a scoring heuristic I invented myself.
        #I created it because it slightly rewards larger icons being fit but significantly rewards higher matching ratio.
        #We want to slightly reward larger icons because smaller icon matching with same percentage might be a partial match and this rewards a full match.
        #The partial match would be for if the larger icon contains the identical smaller icon in it as part of it.
        #However, larger icons will have more random cross over so we have to heavily penalize extra pixels that don't contribute to the vote.
        #This scoring algorithm strikes a very nice balance of slightly rewarding larger icons while heavily penalizing surplus pixels that do not contribute.
        #That's exactly what we want.
        scores = [vote**(vote/max_votes[index]) for index, vote in enumerate(votes[0,:,0])]
        
        #scores = [int(vote) for vote in votes[0,:,0]]
        score_threshold = 0
        nms_threshold = 0.3
        
        indices = cv2.dnn.NMSBoxes(
        boxes,
        scores,
        score_threshold,
        nms_threshold
        )

        #readjust indices to apply non maximum suppression
        detections = [detections[index] for index in indices]
        width = [width[index] for index in indices]
        height = [height[index] for index in indices]
        coords = [coords[index] for index in indices]

    return img_edges, detections, coords, height, width

def over_write_icons(image, detections, coords, height, width, color, dilation, dilation_iterations):

    for i in range(len(coords)):
                
        x = int(round(detections[i][0]))
        y = int(round(detections[i][1]))

        #could use old mask and exclude coords based on permitted values
        
        coord = coords[i]
        if dilation != 1:
            template = np.zeros((height[i], width[i]))
            template[coord[0], coord[1]] = 255

            kernel = np.ones(dilation, np.uint8)

            # Apply dilation
            dilated = cv2.dilate(template, kernel, iterations=dilation_iterations)


            coord = np.where(dilated)
        
        y_coord = coord[0].copy()
        x_coord = coord[1].copy()

        y_coord += int(y - height[i] / 2)
        x_coord += int(x - width[i] / 2)

        
        y_coord = y_coord[y_coord >= 0]
        x_coord = x_coord[y_coord >= 0]

        x_coord = x_coord[y_coord <= image.shape[0]]
        y_coord = y_coord[y_coord <= image.shape[0]]

        y_coord = y_coord[x_coord >= 0]
        x_coord = x_coord[x_coord >= 0]

        y_coord = y_coord[x_coord <= image.shape[1]]
        x_coord = x_coord[x_coord <= image.shape[1]]
        
        coord = np.array([y_coord,x_coord])

        image[coord[0], coord[1]] = color

    return image

def box_icons(detections, width, height, image):
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

def center_point_icons(detections, width, height, image):
    for i, det in enumerate(detections):
        x = int(round(det[0]))
        y = int(round(det[1]))
        cv2.circle(
            image,
            (x, y),
            4,
            (255, 0, 255),
            -1
        )
    
#current images use 100% zoom and 0.22 vote_fraction
vote_fraction = 0.25 #fraction of overlapping pixels required for a positive result

threshold_value = 30

for i, PandID_path in enumerate(PandID_paths):

    PandID_image = cv2.imread(PandID_path)

    PandID_gray = cv2.cvtColor(PandID_image, cv2.COLOR_BGR2GRAY)
    ret, PandID_edges = cv2.threshold(PandID_gray, threshold_value, 255, cv2.THRESH_BINARY_INV) #ret is a place holder return that isn't used anywhere else

    boxed_image = PandID_image.copy()
    PandID_edges, detections, coords, height, width = box_detection(key_paths, boxed_image, PandID_edges, vote_fraction, threshold_value)

    box_icons(detections, width, height, boxed_image)

    color = (0, 0, 255)
    dilation = 1
    dilation_iterations = None
    boxed_image = over_write_icons(boxed_image, detections, coords, height, width, color, dilation, dilation_iterations)

    center_point_icons(detections, width, height, boxed_image)

    img_edges = PandID_edges
    image = boxed_image

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

    if detect_text:
        dilation_iterations = 1
        dilation = 10
        color = (255, 255, 255)
        removed_icons_PandID_image = over_write_icons(PandID_image.copy(), detections, coords, height, width, color, dilation, dilation_iterations)

        cv2.waitKey(0)
        cv2.destroyAllWindows()
        print(pytesseract.image_to_string(Image.fromarray(removed_icons_PandID_image)))
        print("rotated")
        PandID_image_rotated = cv2.rotate(PandID_image, cv2.ROTATE_90_CLOCKWISE)

        cv2.waitKey(0)
        cv2.destroyAllWindows()
        print(pytesseract.image_to_string(Image.fromarray(PandID_image_rotated)))
        cv2.imwrite(fr"C:\Users\admin\Documents\Python\PandID results\text_test_{i}.jpg", removed_icons_PandID_image)

    cv2.imwrite(fr"C:\Users\admin\Documents\Python\PandID results\test_{i}.jpg", image)

############################
#text detection using neural networks
#https://tesseractocr.org/
#https://pypi.org/project/pytesseract/
#https://www.geeksforgeeks.org/python/convert-pdf-to-image-using-python/
#https://github.com/oschwartz10612/poppler-windows/releases/
