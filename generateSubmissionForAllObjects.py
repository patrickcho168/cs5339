import json
from pprint import pprint
import cv2
from keras.models import load_model

TEST_BB_FILENAME = "test_stg1.txt"
TEST_IMAGE_FOLDER = "test_stg1/dummy/"
TEST2_BB_FILENAME = "test_stg2.txt"
TEST2_IMAGE_FOLDER = "test_stg2/"
SIZE_OF_BOX_USED = (64,64)
MODEL_FILENAMES = ["modelALLOBJ_0-64.h5", "modelALLOBJ_1-64.h5", "modelALLOBJ_2-64.h5"]

def merge_several_folds_mean(data, nfolds):
    a = np.array(data[0])
    for i in range(1, nfolds):
        a += np.array(data[i])
    a /= nfolds
    return a.tolist()

def getResult(bbFilename, imgFolder, stage):
    print ("Generating Result")

    # GENERATE MODELS
    models = []
    for modelName in MODEL_FILENAMES:
        model = load_model(modelName)
        models.append(model)

    ids = []
    with open(bbFilename) as data_file:
        data = json.load(data_file)
    for oneImage in data:
        oneImageFilename = imgFolder + oneImage["filename"]
        if stage == 0:
            ids.append(oneImage["filename"])
        else:
            ids.append(oneImageFilename)
        print ("Processing: " + oneImageFilename)
        img = cv2.imread(oneImageFilename)
        h, w, c = img.shape
        numBoxesProcessed = 0
        test_data = []
        X_test_id = []
        for box in oneImage["annotations"]:
            y = box['y']
            x = box['x']
            width = box['width']
            height = box['height']
            if x < 0:
                x = 0
            if y < 0:
                y = 0
            if width > w:
                width = w
            if height > h:
                height = h
            crop_img = img[y:y+height, x:x+width] # Crop from x, y, w, h -> 100, 200, 300, 400
            resized = cv2.resize(crop_img, SIZE_OF_BOX_USED, cv2.INTER_LINEAR)
            test_data.append(resized)
            X_test_id.append(numBoxesProcessed)
            # cv2.imshow("title", resized)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            numBoxesProcessed += 1
        test_data = np.array(test_data, dtype=np.uint8)
        test_data = test_data.transpose((0, 3, 1, 2))
        test_data = test_data.astype('float32')
        test_data = test_data / 255
        batch_size = 20
        yfull_test = []
        test_id = []
        nfolds = len(models)
        for i in range(nfolds):
            model = models[i]
            test_prediction = model.predict(test_data, batch_size=batch_size, verbose=2)
            yfull_test.append(test_prediction)
        test_res = merge_several_folds_mean(yfull_test, nfolds)
        for oneBoxSol in test_res:
            print oneBoxSol

        break
    return predictions, ids

def main():
    prediction1, id1 = getResult(TEST_BB_FILENAME, TEST_IMAGE_FOLDER, 0)
    # prediction2, id2 = getResult(TEST2_BB_FILENAME, TEST_IMAGE_FOLDER, 1)

if __name__ == '__main__':
    main()