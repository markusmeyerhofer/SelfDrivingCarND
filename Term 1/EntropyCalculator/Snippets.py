
img = self.remove_alpha(img)
if self.X_scaler is not None:

    # scale to 64x64 if needed
    if (img.shape[0], img.shape[1]) != (64, 64): img=cv2. resize(img, (64, 64), interpolation=cv2.INTER_NEAREST)

    features = self.extract_image_features(img)
    features = self.X_scaler.transform(np.array(features).reshape(1, -1))
    return self.SVC.predict(features)
else:
    print("No SVC or X_scaler loaded/trained yet.")