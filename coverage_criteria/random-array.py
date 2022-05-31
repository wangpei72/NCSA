import numpy as np

np.set_printoptions(threshold=np.inf)


X_test = np.load("../data/compas-aif360preproc-done/features-test.npy")
Y1_test = np.load("../data/compas-aif360preproc-done/labels-test.npy")
Y2_test = np.load("../data/compas-aif360preproc-done/2d-labels-test.npy")
X_test_h1 = np.hstack((X_test, Y1_test))
X_test_full = np.hstack((X_test_h1, Y2_test))

# relative
for i in range(20):
    for k in range(5):
        row_rand_array = np.arange(X_test_full.shape[0])
        np.random.shuffle(row_rand_array)
        m = int(0.05 * (i + 1) * X_test_full.shape[0])
        x_test_ra_part = X_test_full[row_rand_array[0:m]]

        col_features = np.arange(X_test.shape[1])
        col_labels1 = [X_test.shape[1]]
        col_labels2 = [X_test.shape[1] + 1, X_test.shape[1] + 2]

        x_test_features = x_test_ra_part[:, col_features]
        x_test_label1 = x_test_ra_part[:, col_labels1]
        x_test_label2 = x_test_ra_part[:, col_labels2]

        np.save("../data/compas-aif360preproc-done/test/" + str(k + 1) + "/featrures-test-" + str((i + 1) * 5) + "%.npy",
                x_test_features)
        np.save("../data/compas-aif360preproc-done/test/" + str(k + 1) + "/labels-test-" + str((i + 1) * 5) + "%.npy",
                x_test_label1)
        np.save("../data/compas-aif360preproc-done/test/" + str(k + 1) + "/2d-labels-test-" + str((i + 1) * 5) + "%.npy",
                x_test_label2)

# absolute
for i in range(20):
    for k in range(5):
        row_rand_array = np.arange(X_test_full.shape[0])
        np.random.shuffle(row_rand_array)
        m = int((i + 1) * 100)
        x_test_ra_part = X_test_full[row_rand_array[0:m]]

        col_features = np.arange(X_test.shape[1])
        col_labels1 = [X_test.shape[1]]
        col_labels2 = [X_test.shape[1] + 1, X_test.shape[1] + 2]

        x_test_features = x_test_ra_part[:, col_features]
        x_test_label1 = x_test_ra_part[:, col_labels1]
        x_test_label2 = x_test_ra_part[:, col_labels2]

        np.save("../data/compas-aif360preproc-done/test/" + str(k + 1) + "/featrures-test-" + str(m) + ".npy",
                x_test_features)
        np.save("../data/compas-aif360preproc-done/test/" + str(k + 1) + "/labels-test-" + str(m) + ".npy",
                x_test_label1)
        np.save("../data/compas-aif360preproc-done/test/" + str(k + 1) + "/2d-labels-test-" + str(m) + ".npy",
                x_test_label2)