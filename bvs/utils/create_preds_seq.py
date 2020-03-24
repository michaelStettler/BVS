import numpy as np
import math
import cv2


def create_multi_frame(filters, num_row, num_column, size_image, border=5):
    # compute size of frame
    width_img = num_column * size_image[0] + (num_column - 1) * border
    height_img = num_row * size_image[1] + (num_row - 1) * border
    multi_frame = np.zeros((height_img, width_img, 3))

    # stack images into the frame
    for r in range(num_row):
        for c in range(num_column):
            startX = c * (size_image[0] + border)
            stopX = startX + size_image[0]
            startY = r * (size_image[1] + border)
            stopY = startY + size_image[1]

            filter = filters[:, :, :, r * num_column + c]
            filter = (filter - np.min(filter))
            filter = filter / np.max(filter)
            filter = np.array(filter * 255).astype(np.uint8)
            filter = cv2.resize(filter, (256, 256))

            multi_frame[startY:stopY, startX:stopX, :] = filter

    return np.array(multi_frame).astype(np.uint8)


def create_multi_frame_heatmap(image, filters, num_row, num_column, size_image, border=5):
    alpha = 0.25

    # compute size of frame
    width_img = num_column * size_image[0] + (num_column - 1) * border
    height_img = num_row * size_image[1] + (num_row - 1) * border
    multi_frame = np.zeros((height_img, width_img, 3))

    image = image - np.min(image)
    image = np.array((image / np.max(image)) * 255).astype(np.uint8)

    # stack images into the frame
    for r in range(num_row):
        for c in range(num_column):
            startX = c * (size_image[0] + border)
            stopX = startX + size_image[0]
            startY = r * (size_image[1] + border)
            stopY = startY + size_image[1]

            filter = filters[:, :, r * num_column + c]
            filter = (filter - np.min(filter))
            filter = filter / np.max(filter)
            filter = np.array(filter * 255).astype(np.uint8)
            filter = cv2.resize(filter, (256, 256))

            # heatmap = cv2.applyColorMap(filter, cv2.COLORMAP_HOT)
            heatmap = cv2.applyColorMap(filter, cv2.COLORMAP_VIRIDIS)
            output = cv2.addWeighted(image, alpha, heatmap, 1 - alpha, 0)

            multi_frame[startY:stopY, startX:stopX, :] = output

    return np.array(multi_frame).astype(np.uint8)


def create_preds_seq(data, preds, max_column=4):
    # declare variables
    border = 5  # px

    # get dims
    length_seq = np.shape(data)[0]
    num_images = np.shape(preds)[-1]
    size_image = (np.shape(data)[1], np.shape(data)[2])

    # stack images together
    num_column = min(num_images, max_column)
    num_row = math.ceil(num_images / max_column)

    # declare array
    width_img = num_column * size_image[0] + (num_column - 1) * border
    height_img = num_row * size_image[1] + (num_row - 1) * border
    seq = np.zeros((length_seq, height_img, width_img, 3))
    print("shape seq", np.shape(seq))

    # create sequence
    for s in range(length_seq):
        img = data[s]
        img = img - np.min(img)
        img = np.array((img / np.max(img)) * 255).astype(np.uint8)

        for r in range(num_row):
            for c in range(num_column):
                startX = c * (size_image[0] + border)
                stopX = startX + size_image[0]
                startY = r * (size_image[1] + border)
                stopY = startY + size_image[1]

                filter = preds[s, :, :, r*num_column + c]
                filter = (filter - np.min(filter))
                filter = filter / np.max(filter)
                filter = np.array(filter * 255).astype(np.uint8)
                filter = cv2.resize(filter, (256, 256))

                alpha = 0.25
                # heatmap = cv2.applyColorMap(filter, cv2.COLORMAP_HOT)
                heatmap = cv2.applyColorMap(filter, cv2.COLORMAP_VIRIDIS)
                output = cv2.addWeighted(img, alpha, heatmap, 1 - alpha, 0)

                seq[s, startY:stopY, startX:stopX, :] = output

        # cv2.imshow("test", seq[s].astype(np.uint8))
        # cv2.waitKey(0)

    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    writer = cv2.VideoWriter('test1.avi', fourcc, 30, size_image, True)
    for i in range(length_seq):
        writer.write(seq[i].astype(np.uint8))
        cv2.imwrite("bvs/video/seq_"+str(i)+".jpeg", seq[i].astype(np.uint8))

    writer.release()
