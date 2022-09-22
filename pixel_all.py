# Python imports
import sys
from datetime import datetime

# Third party imports
import cv2
import numpy as np


# Import the image from system arg
print("Importing image")
img_file = sys.argv[1]
raw_img = cv2.imread(img_file)
input_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)

# Convert the numpy coordinates into pixel art with the right colors
def img_to_pixels(coord_pixels, pixel_mod):
    print("Generating pixel image.")

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

            # Generate a mock pixel that is n x n "real" pixels
            for size_p in range(1, size_pixel + 1):
                for l_p in range(0, size_p):
                    np_array[y - size_p][x - size_p + l_p] = c
                    np_array[y + size_p - size_pixel - 1][
                        x + size_p - l_p - size_pixel - 1
                    ] = c

        # Save the outfile file and crop 10 pixels from the edge
        filename = (
            "Mod: "
            + str(int(pixel_mod * 100))
            + " "
            + "Pixel Size: "
            + str(size_pixel)
            + ".png"
        )
        img_cropped = np_array[10 : np_array.shape[0] - 10, 10 : np_array.shape[1] - 10]
        cv2.imwrite(filename, img=img_cropped)
        print("Completed - %s" % (filename))


# Creates an array of tuples of x and y values that represent the cartesian plane in positive
pix_x = list(np.arange(0, raw_img.shape[1], 1, dtype=float))
pix_y = list(np.arange(0, raw_img.shape[0], 1, dtype=float))
pixel_list = [(y, x) for y in pix_y for x in pix_x]
coord_pixels = np.array(pixel_list)

# Modifier function to control output pixel density
pixel_mods = [0.05, 0.1]

# Itterate the pixel modulations to generate images for each pixel density
for pixel_mod in pixel_mods:
    print("\nGenerating image with pixel modulation: %s" % (pixel_mod))

    # New coordinate lists
    new_coord = []
    mod_coord = []

    # Variables to count progress
    amount_p = input_img.shape[0] * input_img.shape[1]
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

        # Modulate each pixel
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
    print("Generated data points with pixel modulation %s." % (pixel_mod))
    img_to_pixels(mod_coord, pixel_mod)

# Destroy all windows to close
cv2.destroyAllWindows()
