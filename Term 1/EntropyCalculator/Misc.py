def combine_and_normalize(self):
    if len(self.car_features) > 0:
        self.y = np.hstack((np.ones(len(self.car_features)), np.zeros(len(self.non_car_features))))
        # Create an array stack of feature vectors
        print
        self.X = np.vstack((self.car_features, self.non_car_features)).astype(np.float64)
        # Fit a per-column scaler
        self.X_scaler = StandardScaler().fit(self.X)
        # Apply the scaler to X
        self.scaled_X = self.X_scaler.transform(self.X)
    return


def save_classifier(self):
    # save the classifier
    # joblib.dump(self.SVC, 'SVC-Classifier.pkl')
    print("Saving Classifier...")
    svc_bin = {"svc": self.SVC, "scaler": self.X_scaler}
    pickle.dump(svc_bin, open("svc_pickle.p", "wb"))
    print("Classifier Saved.")


def load_classifier(self):
    if os.path.isfile('svc_pickle.p'):
        print("Loading Classifier...")
        # load it again
        # self.SVC = joblib.load('SVC-Classifier.pkl')

        dist_pickle = pickle.load(open("svc_pickle.p", "rb"))
        self.SVC = dist_pickle["svc"]
        self.X_scaler = dist_pickle["scaler"]
        print("Classifier Loaded.")
    else:
        print("No preprocessed classifier available yet. Preprocessing ...")
        self.preprocess()
        print("Preprocessing completed.")


def train(self):
    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(self.scaled_X, self.y, test_size=0.1, random_state=rand_state)

    print('Using:', 32, 'orientations', 32, 'pixels per cell and', 2, 'cells per block')
    print('Feature vector length:', len(X_train[0]))
    # Use a linear SVC
    self.SVC = LinearSVC()
    # Check the training time for the SVC
    import time
    t = time.time()
    self.SVC.fit(X_train, y_train)
    t2 = time.time()
    self.save_classifier()
    print(round(t2 - t, 2), 'Seconds to train SVC...')
    # Check the score of the SVC
    print('Test Accuracy of SVC = ', round(self.SVC.score(X_test, y_test), 4))
    # Check the prediction time for a single sample
    t = time.time()
    n_predict = 100
    print('My SVC predicts: ', self.SVC.predict(X_test[0:n_predict]))
    print('For these', n_predict, 'labels: ', y_test[0:n_predict])
    t2 = time.time()
    print(round(t2 - t, 5), 'Seconds to predict', n_predict, 'labels with SVC')


def preprocess(self):
    self.car_features = self.extract_features(self.car_images)
    self.non_car_features = self.extract_features(self.non_car_images)
    self.combine_and_normalize()
    self.train()


def predict_image(self, img):
    if self.X_scaler is not None:
        img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_NEAREST)
        features = self.extract_image_features(img)
        features = self.X_scaler.transform(np.array(features).reshape(1, -1))
        return self.SVC.predict(features)
    else:
        print("No SVC or X_scaler loaded/trained yet.")


def convert_color(self, img, conv='RGB2YCrCb'):
    if conv == 'RGB2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    if conv == 'RGB2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    if conv == 'RGB2LUV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2LUV)


def print_bboxes(self, img, bboxes, print=False):
    for box in bboxes:
        cv2.rectangle(img, (box[0][0], box[0][1]), (box[1][0], box[1][1]), (0, 0, 255), 6)

    if print == True:
        plt.figure(figsize=(30, 10))
        plt.subplot(1, 2, 1)
        plt.hold(True)
        plt.imshow(img)


def find_cars(self, img, scale=1.5):
    bboxes = []
    draw_img = np.copy(img)
    # img = img.astype(np.float32) / 255

    img_tosearch = img[self.ystart:self.ystop, :, :]
    ctrans_tosearch = self.convert_color(img_tosearch, conv='RGB2YCrCb')
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1] / scale), np.int(imshape[0] / scale)))

    ch1 = ctrans_tosearch[:, :, 0]
    ch2 = ctrans_tosearch[:, :, 1]
    ch3 = ctrans_tosearch[:, :, 2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // self.pix_per_cell) - 1
    nyblocks = (ch1.shape[0] // self.pix_per_cell) - 1
    nfeat_per_block = self.orient * self.cell_per_block ** 2
    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // self.pix_per_cell) - 1
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step

    # Compute individual channel HOG features for the entire image
    hog1 = self.get_hog_features(ch1, feature_vec=False)
    hog2 = self.get_hog_features(ch2, feature_vec=False)
    hog3 = self.get_hog_features(ch3, feature_vec=False)

    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb * cells_per_step
            xpos = xb * cells_per_step
            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_feat2 = hog2[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_feat3 = hog3[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

            xleft = xpos * self.pix_per_cell
            ytop = ypos * self.pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop + window, xleft:xleft + window], (64, 64))

            # Get color features
            spatial_features = self.bin_spatial(subimg)
            hist_features = self.color_hist(subimg)

            # Scale features and make a prediction
            test_features = self.X_scaler.transform(
                np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))
            # test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))
            test_prediction = self.SVC.predict(test_features)

            box = []
            if test_prediction == 1:
                xbox_left = np.int(xleft * scale)
                ytop_draw = np.int(ytop * scale)
                win_draw = np.int(window * scale)
                box = ((xbox_left, ytop_draw + self.ystart), (xbox_left + win_draw, ytop_draw + win_draw + self.ystart))
                bboxes.append(box)

    self.print_bboxes(draw_img, bboxes)

    return bboxes