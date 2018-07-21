from functions import *

def process_image(image):
    global y_start_stop, svc, X_scaler, color_space, spatial_size, hist_bins, orient, pix_per_cell, cell_per_block, hog_channel, spatial_feat, hist_feat, hog_feat
    #image_copy = np.copy(image)
    image_copy = image.astype(np.float32) / 255
    draw_image = np.copy(image_copy)

    windows = average_slide_windows(image, x_start_stop=[None, None], y_start_stop=y_start_stop,
                           xy_window=(80, 80), xy_overlap=(0.8, 0.8))

    hot_windows = search_windows(image_copy, windows, svc, X_scaler, color_space=color_space,
                                  spatial_size=spatial_size, hist_bins=hist_bins,
                                  orient=orient, pix_per_cell=pix_per_cell,
                                  cell_per_block=cell_per_block,
                                  hog_channel=hog_channel, spatial_feat=spatial_feat,
                                  hist_feat=hist_feat, hog_feat=hog_feat)


    # hot_windows.append(np.concatenate((hot_windows1, hot_windows2), axis=0))
    # hot_windows = np.squeeze(hot_windows)

    window_img = draw_boxes(draw_image, hot_windows, color=(0, 0, 255), thick=6)

    heat = np.zeros_like(window_img[:,:,0]).astype(np.float)
    heat = add_heat(heat,hot_windows)
    heatmap = apply_threshold(heat,1)
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
    # Reduce the sample size
    #sample_size = 500
    #cars = cars[0:sample_size]
    #notcars = notcars[0:sample_size]
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

    car_features, notcar_features = get_features(cars, notcars, color_space, orient, pix_per_cell, cell_per_block,
                                                 hog_channel, spatial_size, hist_bins, spatial_feat,
                                                 hist_feat, hog_feat)

    svc = train_classifier(car_features, notcar_features, spatial_size, hist_bins)

    image = mpimg.imread('../test_images/test5.jpg')
    image = image.astype(np.float32)/255
    draw_image = np.copy(image)

    X = np.vstack((car_features, notcar_features)).astype(np.float64)
    X_scaler = StandardScaler().fit(X)
    scaled_X = X_scaler.transform(X)

    windows = average_slide_windows(image, x_start_stop=[None, None], y_start_stop=y_start_stop,
                                    xy_window=(80, 80), xy_overlap=(0.8, 0.8))

    hot_windows = search_windows(image, windows, svc, X_scaler, color_space=color_space,
                            spatial_size=spatial_size, hist_bins=hist_bins,
                            orient=orient, pix_per_cell=pix_per_cell,
                            cell_per_block=cell_per_block,
                            hog_channel=hog_channel, spatial_feat=spatial_feat,
                            hist_feat=hist_feat, hog_feat=hog_feat)

    #Show heat map
    show_heat_image(image, hot_windows)

    window_img = draw_boxes(draw_image, hot_windows, color=(110, 240, 41), thick=6)

    plt.imshow(window_img)
    plt.imsave("../output_images/test5.jpg", window_img)

    heat = np.zeros_like(image[:, :, 0]).astype(np.float)
    # Add heat to each box in box list
    heat = add_heat(heat, hot_windows)

    # Apply threshold to help remove false positives
    heatmap = apply_threshold(heat, 1)

    # Visualize the heatmap when displaying
    # heatmap = np.clip(heat, 0, 255)

    # Find final boxes from heatmap using label function
    labels = label(heatmap)
    draw_img = draw_labeled_bboxes(np.copy(image), labels)
    plt.imshow(draw_img)
    plt.imsave("../output_images/test5_heat.jpg", draw_img)



    image = mpimg.imread('../test_images/test4.jpg')
    result = process_image(image)
    plt.imshow(result)

    video_output = '../output_images/project_video.mp4'
    clip1 = VideoFileClip("../project_video.mp4", audio=False)

    video_clip = clip1.fl_image(process_image)
    video_clip.write_videofile(video_output, audio=False)