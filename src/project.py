from functions import *
import os.path
import pickle
from collections import deque

heatmaps = deque(maxlen = 25) #You can choose how many frames to use.
heatmap_threshold = 1

def process_image(image):
    global y_start_stop, svc, X_scaler, color_space, spatial_size, hist_bins, orient, pix_per_cell, cell_per_block, hog_channel, spatial_feat, hist_feat, hog_feat, heatmaps

    window_img, hot_windows = find_cars(image, y_start_stop[0], y_start_stop[1], 1.5, svc, X_scaler, orient, pix_per_cell,
                                        cell_per_block, spatial_size, hist_bins)


    heat = np.zeros_like(window_img[:,:,0]).astype(np.float)
    heat = add_heat(heat,hot_windows)

    heatmaps.append(heat)
    combined = sum(heatmaps)/len(heatmaps)

    heatmap = apply_threshold(combined,heatmap_threshold)
    labels = label(heatmap)
    result = draw_labeled_bboxes(np.copy(image), labels)

    return result

if __name__ == "__main__":

    # Read in car and non-car images
    car_images = glob.glob('../training_images/vehicles/*/*.png')
    non_car_images = glob.glob('../training_images/non-vehicles/*/*.png')

    cars = []
    notcars = []
    for image in car_images:
        cars.append(image)
    for image in non_car_images:
        notcars.append(image)

    print("The total images in the car dataset are:",len(cars))
    print("The total images in the not-car dataset are:",len(notcars))

    # Show examples of car / non-car training images
    show_car_notcat_samples(cars, notcars)

    show_hog_sample(mpimg.imread('../training_images/vehicles/GTI_MiddleClose/image0002.png'))


    color_space = 'YCrCb'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
    orient = 9  # HOG orientations
    pix_per_cell = 8  # HOG pixels per cell
    cell_per_block = 2  # HOG cells per block
    hog_channel = "ALL"  # Can be 0, 1, 2, or "ALL"
    spatial_size = (16, 16)  # Spatial binning dimensions
    hist_bins = 32  # Number of histogram bins
    spatial_feat = True  # Spatial features on or off
    hist_feat = True  # Histogram features on or off
    hog_feat = True  # HOG features on or off
    y_start_stop = [340, 680]  # Min and max in y to search in slide_window()
    xy_overlap = (0.8, 0.8)
    xy_window = [64,64]
    x_start_stop = [760, 1260]

    filename1 = './standard_scaler.sav'
    filename2 = './svc.sav'
    image = mpimg.imread('../test_images/test5.jpg')

    if not os.path.isfile(filename1) or not os.path.isfile(filename2):
        car_features, notcar_features = get_features(cars, notcars, color_space, orient, pix_per_cell, cell_per_block,
                                                     hog_channel, spatial_size, hist_bins, spatial_feat,
                                                     hist_feat, hog_feat)
        svc = train_classifier(car_features, notcar_features, spatial_size, hist_bins)

        draw_image = np.copy(image)

        X = np.vstack((car_features, notcar_features)).astype(np.float64)
        model = StandardScaler()
        X_scaler = model.fit(X)

        pickle.dump(X_scaler, open(filename1, 'wb'))
        pickle.dump(svc, open(filename2, 'wb'))


    else:
        X_scaler = pickle.load(open(filename1, 'rb'))
        svc = pickle.load(open(filename2, 'rb'))


    window_img, hot_windows = find_cars(image, y_start_stop[0], y_start_stop[1], 1.5, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)


    plt.imshow(window_img)
    plt.imsave("../output_images/test5.jpg", window_img)

    heat = np.zeros_like(image[:, :, 0]).astype(np.float)
    # Add heat to each box in box list
    heat = add_heat(heat, hot_windows)
    heatmaps.append(heat)
    combined = sum(heatmaps)/len(heatmaps)
    # Apply threshold to help remove false positives
    heatmap = apply_threshold(combined, heatmap_threshold)

    # Visualize the heatmap when displaying
    # heatmap = np.clip(heat, 0, 255)

    # Find final boxes from heatmap using label function
    labels = label(heatmap)
    draw_img = draw_labeled_bboxes(np.copy(image), labels)
    plt.imshow(draw_img)
    plt.imsave("../output_images/test5_final.jpg", draw_img)



    image = mpimg.imread('../test_images/test4.jpg')
    result = process_image(image)
    plt.imsave("../output_images/test4_final.jpg", result)

    video_output = '../output_images/project_video.mp4'
    clip1 = VideoFileClip("../project_video.mp4", audio=False)

    video_clip = clip1.fl_image(process_image)
    video_clip.write_videofile(video_output, audio=False)