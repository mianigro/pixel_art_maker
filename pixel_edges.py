# Python imports
import sys
from datetime import datetime

# Third party imports
import cv2
import numpy as np


# Import the image from system arg
img_file = sys.argv[1]
raw_img = cv2.imread(img_file)


# Used for canny bars
def empty_function(*args):
    pass


# Detect the edges and out the new image as a numpy array
def convert_canny(img):

    # Define the dinwo and the track bars
    win_name = "Threshold Selection"
    cv2.namedWindow(win_name)
    cv2.resizeWindow(win_name, 200, 200)
    cv2.createTrackbar("canny_th1", win_name, 0, 255, empty_function)
    cv2.createTrackbar("canny_th2", win_name, 0, 255, empty_function)

    # Display the image
    while True:

        # Add the track bars
        cth1_pos = cv2.getTrackbarPos("canny_th1", win_name)
        cth2_pos = cv2.getTrackbarPos("canny_th2", win_name)

        # Process images
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_blur = cv2.GaussianBlur(img_gray, (3, 3), 1)

        # Get edges using the trackbars
        img_canny = cv2.Canny(img_blur, cth1_pos, cth2_pos)

        # SHowing the image and setting exit key
        cv2.imshow(win_name, img_canny)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("c"):
            cv2.destroyAllWindows()
            break

    # Return the edges of the image
    return img_canny


# Convert the numpy coordinates into pixel art with the right colors
def make_pixel_edges(coord_pixels, pixel_mod):

    # Make an array that is the dimensions of the original image with color chanel
    np_array = np.full(raw_img.shape, 200, dtype="uint8")

    # Pixel size, n x n
    size_pixels = [3, 5, 7, 9]

    # Loops through the pixel sizes
    for size_pixel in size_pixels:

        # Loops through the coordinates of pixels
        for y, x in coord_pixels:

            # Gets the color of the pixel on the original image at this location
            c = raw_img[y, x]

            # Generate a mock pixel that is n x n pixels
            for size_p in range(1, size_pixel + 1):
                for l_p in range(0, size_p):
                    np_array[y - size_p][x - size_p + l_p] = c
                    np_array[y + size_p - size_pixel - 1][
                        x + size_p - l_p - size_pixel - 1
                    ] = c

        # Save the outfile file
        filename = (
            "Mod: "
            + str(int(pixel_mod * 100))
            + " "
            + "Pixel Size: "
            + str(size_pixel)
            + ".png"
        )
        cv2.imwrite(filename, img=np_array)


# Convert raw img into canny image
print("Edge detection on image commencing.")
canny_img_out = convert_canny(raw_img)


# Find pixels where not empty, array 1 is x pixels, array 2 is y pixels
pixel_indices = np.where(canny_img_out != [0])


# Using zip function to combine the arrays into a list of coors for each pixel
coord_pixels = list(zip(pixel_indices[0], pixel_indices[1]))


# Modifier function to control output pixel density
pixel_mods = [0.05, 0.1, 0.15, 0.2]


# Itterate the pixel modulations to generate images for each pixel density
for pixel_mod in pixel_mods:
    print("Generating image with pixel modulation: %s" % (pixel_mod))

    # New coordinate lists
    new_coord = []
    mod_coord = []

    # Variables to count progress
    amount_p = int(len(coord_pixels))
    count_p1 = 1
    buffer = 0

    # Turn the pixels into whole numbers using the modulation to filter
    print("Generating pixel coords step 1.")
    for x, y in coord_pixels:

        # Counting logic
        if int(count_p1 / amount_p * 100) % 5 == 0 and buffer != int(
            count_p1 / amount_p * 100
        ):
            current_time = datetime.now().strftime("%H:%M:%S")
            print(
                "%s percent completed at %s"
                % (int(count_p1 / amount_p * 100), current_time)
            )
            buffer = int(count_p1 / amount_p * 100)
        count_p1 += 1

        # Modulate pixel
        new_y = int(y * pixel_mod)
        new_x = int(x * pixel_mod)

        # Filter out duplicates and assign to a list of tuples
        if tuple([new_y, new_x]) not in new_coord:
            new_coord.append(tuple([new_y, new_x]))

    # Variables to count progress
    count_p2 = 1
    amount_p2 = int(len(new_coord))
    buffer2 = 0

    # Send pixel back to original but is filtered
    print("Generating pixel coords step 2.")
    for x, y in new_coord:

        # Counting logic
        if int(count_p2 / amount_p2 * 100) % 5 == 0 and buffer2 != int(
            count_p2 / amount_p2 * 100
        ):
            current_time = datetime.now().strftime("%H:%M:%S")
            print(
                "%s percent completed at %s"
                % (int(count_p2 / amount_p2 * 100), current_time)
            )
            buffer2 = int(count_p2 / amount_p2 * 100)
        count_p2 += 1

        # Resize pixels to original size
        mod_y = int(y / pixel_mod)
        mod_x = int(x / pixel_mod)

        # Check for duplicates and assign to a list of tuples
        if tuple([mod_y, mod_x]) not in mod_coord:
            mod_coord.append(tuple([mod_y, mod_x]))

    # Generating the pixel image based of the filtered data
    print("Saving pixel art with pixel modulation %s. \n" % (pixel_mod))
    make_pixel_edges(mod_coord, pixel_mod)

# Destroy all windows to close
cv2.destroyAllWindows()
