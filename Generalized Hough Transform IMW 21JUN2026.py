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
path_pattern = r"C:\Users\admin\Documents\Python\Hough PandID"
PandID_paths = glob.glob(path_pattern + r"/**/*", recursive=True)
#print(PandID_paths)

path_pattern = r"C:\Users\admin\Documents\Python\Hough key"
key_paths = glob.glob(path_pattern + r"/**/*", recursive=True)
#print(key_paths)

# Load images
#template = cv2.imread("valve image.jpg", cv2.IMREAD_GRAYSCALE)

#Comment out if don't want to rotate
#template = cv2.rotate(template, cv2.ROTATE_90_CLOCKWISE)

def box_detection(template_paths, image, img_edges, vote_fraction, threshold_value):
    #image = cv2.imread("PandID with valves.jpg")

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




        for i in range(4):
            templ_edges = cv2.rotate(templ_edges, cv2.ROTATE_90_CLOCKWISE)
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

            coord = np.where(templ_edges)
            
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
                #print(max(coord[0]))
                coords += [coord]*positions_rotation.shape[1]
                #if votes_rotation is not None:
                votes += [votes_rotation]
                max_votes += [max_possible_votes]*positions_rotation.shape[1]

    if positions != []:
        #print(positions)
        #print(votes)
        #print(positions)
        positions = np.concatenate(positions, axis = 1)
        #print(positions)
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
        #print(indices)

        #readjust indices to apply non maximum suppression
        detections = [detections[index] for index in indices]
        width = [width[index] for index in indices]
        height = [height[index] for index in indices]
        coords = [coords[index] for index in indices]

        #print(len(width))
        #print(len(height))
        #print(len(coords))
        #print(list(zip(height, width)))
        #print([m.shape for m in mask])

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
            """
            #calculate edges of mask
            x1_mask = int(x - width[i] / 2)
            x2_mask = int(x + width[i] / 2)
            y1_mask = int(y - height[i] / 2)
            y2_mask = int(y + height[i] / 2)

            #calculate edges of image
            x1_image = 0
            y1_image = 0
            x2_image = image.shape[1]
            y2_image = image.shape[0]

            #calculate edges of new mask overlays in image coordinates
            x1_mask_permitted = max(x1_mask,x1_image)
            #print("before")
            x2_mask_permitted = min(x2_mask,x2_image)
            #print("after")
            y1_mask_permitted = max(y1_mask,y1_image)
            y2_mask_permitted = min(y2_mask,y2_image)

            #calculates edges of new mask in old mask coordinates
            x1_mask_adjustment = x1_mask_permitted - x1_mask
            x2_mask_adjustment = x2_mask_permitted - x2_mask
            y1_mask_adjustment = y1_mask_permitted - y1_mask
            y2_mask_adjustment = y2_mask_permitted - y2_mask

            #print(x1_mask_adjustment)
            #print(x2_mask_adjustment)
            #print(y1_mask_adjustment)
            #print(y2_mask_adjustment)
            #print(w)
            #print(h)
            #print(w + x2_mask_adjustment)
            #print(h + y2_mask_adjustment)
            #print(mask[i].shape)

            #print(y1_mask_permitted)
            #print(y2_mask_permitted)
            #print(x1_mask_permitted)
            #print(x2_mask_permitted)

            #print(type(mask[i]))
            new_mask = mask[i][y1_mask_adjustment:h + y2_mask_adjustment, x1_mask_adjustment:w + x2_mask_adjustment]
            
            new_image_section = image[y1_mask_permitted:y2_mask_permitted, x1_mask_permitted:x2_mask_permitted]

            #print(new_mask.shape)
            #print(new_image_section.shape)

            #masked_image = cv2.bitwise_and(new_image_section, new_image_section, mask = new_mask)
            #masked_image = new_image_section[new_mask != 0] = (0, 0, 255)
            #new_image_section[new_mask] = (0, 0, 255)
            #masked_image = new_image_section
            color = (0, 0, 255)
            #comment out
            color_layer = np.full_like(new_image_section, color, dtype=np.uint8)

            colored_mask = cv2.bitwise_and(color_layer, color_layer, mask=new_mask)
            new_mask_inverse = cv2.bitwise_not(new_mask)
            image_cutout = cv2.bitwise_and(new_image_section, new_image_section, mask=new_mask_inverse)
            masked_image = cv2.add(image_cutout, colored_mask)
            #masked_image = np.where(new_mask, color_layer, new_image_section)

            image[y1_mask_permitted:y2_mask_permitted, x1_mask_permitted:x2_mask_permitted] = masked_image
            """
        #print(len(detections))
        #print(len(coords))
        """
        for i in range(len(coords)):
            #comment out
            color = (0, 0, 255)
            # Vectorized assignment
            #could use old mask and exclude coords based on permitted values
            #coords = np.where(new_mask)
            coord = coords[i]
            y_coord = coord[0].copy()
            x_coord = coord[1].copy()
            #print(len(x_coord))
            #print(len(y_coord))
            #print(coords[0])
            #print(coords[1])
            #print(int(y - height[i] / 2))
            #print(max(y_coord))
            y_coord += int(y - height[i] / 2)
            x_coord += int(x - width[i] / 2)

            
            y_coord = y_coord[y_coord >= 0]
            x_coord = x_coord[y_coord >= 0]

            #print(len(x_coord))
            #print(len(y_coord))
            #print(max(y_coord))
            x_coord = x_coord[y_coord <= image.shape[0]]
            y_coord = y_coord[y_coord <= image.shape[0]]
            #print(image.shape[0])
            
            #print(len(x_coord))
            #print(len(y_coord))
            #x_coord = x_coord[y_coord <= image.shape[0]]

            y_coord = y_coord[x_coord >= 0]
            x_coord = x_coord[x_coord >= 0]

            y_coord = y_coord[x_coord <= image.shape[1]]
            x_coord = x_coord[x_coord <= image.shape[1]]
            
            coord = np.array([y_coord,x_coord])
            #coords[0] = np.max(coords[0], h)
            #coords[0] = np.min(coords[0], 0)
            #coords[1] = np.max(coords[0], w)
            #coords[1] = np.min(coords[0], 0)

            #print(coord[0])
            #print(coord[1])

            image[coord[0], coord[1]] = color
            #print("pass")
            """
            
            #image_cutout[coords[0], coords[1]] = color
            #image[y1_mask_permitted:y2_mask_permitted, x1_mask_permitted:x2_mask_permitted] = image_cutout

        color = (0, 0, 255)
        dilation = 1
        dilation_iterations = None
        image = over_write_icons(image, detections, coords, height, width, color, dilation, dilation_iterations)
        for i in range(len(coords)):
            # Center point
            cv2.circle(
                image,
                (x, y),
                4,
                (255, 0, 255),
                -1
            )

    return img_edges, detections, coords, height, width

def over_write_icons(image, detections, coords, height, width, color, dilation, dilation_iterations):

    for i in range(len(coords)):
                
        x = int(round(detections[i][0]))
        y = int(round(detections[i][1]))

        #comment out
        #color = (0, 0, 255)
        # Vectorized assignment
        #could use old mask and exclude coords based on permitted values
        #coords = np.where(new_mask)
        coord = coords[i]
        if dilation != 1:
            template = np.zeros((height[i], width[i]))
            template[coord[0], coord[1]] = 255

            kernel = np.ones(dilation, np.uint8)

            # Apply dilation
            dilated = cv2.dilate(template, kernel, iterations=dilation_iterations)


            coord = np.where(dilated)

            # Show the result
            #cv2.imshow("Detected Lines", template)
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()
            
            #print(template.shape)
        
        y_coord = coord[0].copy()
        x_coord = coord[1].copy()
        #print(len(x_coord))
        #print(len(y_coord))
        #print(coords[0])
        #print(coords[1])
        #print(int(y - height[i] / 2))
        #print(max(y_coord))
        y_coord += int(y - height[i] / 2)
        x_coord += int(x - width[i] / 2)

        
        y_coord = y_coord[y_coord >= 0]
        x_coord = x_coord[y_coord >= 0]

        #print(len(x_coord))
        #print(len(y_coord))
        #print(max(y_coord))
        x_coord = x_coord[y_coord <= image.shape[0]]
        y_coord = y_coord[y_coord <= image.shape[0]]
        #print(image.shape[0])
        
        #print(len(x_coord))
        #print(len(y_coord))
        #x_coord = x_coord[y_coord <= image.shape[0]]

        y_coord = y_coord[x_coord >= 0]
        x_coord = x_coord[x_coord >= 0]

        y_coord = y_coord[x_coord <= image.shape[1]]
        x_coord = x_coord[x_coord <= image.shape[1]]
        
        coord = np.array([y_coord,x_coord])
        #coords[0] = np.max(coords[0], h)
        #coords[0] = np.min(coords[0], 0)
        #coords[1] = np.max(coords[0], w)
        #coords[1] = np.min(coords[0], 0)

        #print(coord[0])
        #print(coord[1])

        image[coord[0], coord[1]] = color
        #color_layer = np.full_like(template, color, dtype=np.uint8)
        #colored_mask = cv2.bitwise_and(color_layer, color_layer, mask=template)
        #new_mask_inverse = cv2.bitwise_not(template)
        #image_cutout = cv2.bitwise_and(new_image_section, new_image_section, mask=new_mask_inverse)
        #masked_image = cv2.add(image_cutout, colored_mask)
        #h_2 = int(y+height[i])
        #w_2 = int(x+width[i])
        #image[y:int(y+height[i]), x:int(x+width[i])] = template

    return image
    
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

    PandID_gray = cv2.cvtColor(PandID_image, cv2.COLOR_BGR2GRAY)
    ret, PandID_edges = cv2.threshold(PandID_gray, threshold_value, 255, cv2.THRESH_BINARY_INV) #ret is a place holder return that isn't used anywhere else

    boxed_image = PandID_image.copy()
    PandID_edges, detections, coords, height, width = box_detection(key_paths, boxed_image, PandID_edges, vote_fraction, threshold_value)

    #key_image_90_rotated = cv2.rotate(key_image, cv2.ROTATE_90_CLOCKWISE)
    #PandID_edges = box_detection(key_image_90_rotated, PandID_image, PandID_edges, vote_fraction, threshold_value)

    img_edges = PandID_edges
    image = boxed_image

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

    if detect_text:
        dilation_iterations = 1
        dilation = 10
        color = (255, 255, 255)
        removed_icons_PandID_image = over_write_icons(PandID_image.copy(), detections, coords, height, width, color, dilation, dilation_iterations)
        #cv2.imshow("Ballard Detection", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        print(pytesseract.image_to_string(Image.fromarray(removed_icons_PandID_image)))
        print("rotated")
        PandID_image_rotated = cv2.rotate(PandID_image, cv2.ROTATE_90_CLOCKWISE)
        #cv2.imshow("rotated", PandID_image_rotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        print(pytesseract.image_to_string(Image.fromarray(PandID_image_rotated)))
        cv2.imwrite(fr"C:\Users\admin\Documents\Python\PandID results\text_test_{i}.jpg", removed_icons_PandID_image)

    cv2.imwrite(fr"C:\Users\admin\Documents\Python\PandID results\test_{i}.jpg", image)

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
