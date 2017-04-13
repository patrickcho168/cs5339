import json
import os
import PIL
from PIL import Image

iouThreshold = .5

dirPath = '../annotations'
files = os.listdir(dirPath)

for jsonPath in files:
    
    jsonFile = open(os.path.join(dirPath , jsonPath))
    jsonData = json.load(jsonFile)

    subDir = ''
    dirsList = ['ALB', 'BET', 'DOL', 'LAG', 'SHARK', 'YFT', 'OTHER', 'NoF']
    for dir in dirsList:
        if(dir.lower() in jsonPath.lower()):
            subDir = dir
            break

    #json represent images for a single type of fish
    #Now Iterating over the images for particular type of fish
    for image in jsonData:
        objs = image['annotations']
        
        iterator = 1
        imgDir = os.path.join('..', 'train', subDir, image['filename'])
        print('Loading {}'.format(imgDir))
        im = Image.open(imgDir)
        #Iterate over the objects given in image
        for obj in objs:
            
            savePath = os.path.join( '..', 'croppedImagesGroundTruth', subDir, image['filename'] + '-' + str(iterator) + '.jpg')
            #print('Saving image: {}'.format(savePath))
            im.crop((obj['x'],obj['y'],obj['width'] + obj['x'], obj['height'] + obj['y'])).save(savePath)
            iterator += 1;
