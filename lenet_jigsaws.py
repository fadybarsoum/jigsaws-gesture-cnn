# import the necessary packages
import os
os.environ['MKL_NUM_THREADS'] = '4'
os.environ['GOTO_NUM_THREADS'] = '4'
os.environ['OMP_NUM_THREADS'] = '4'
#os.environ['THEANO_FLAGS'] = 'floatX=float32,openmp=True'
#import theano
#theano.config.openmp=True
from cnn.networks.lenet import LeNet # from jigsawsgesturecnn.
from sklearn.cross_validation import train_test_split
from sklearn import datasets
from skimage.io import ImageCollection
from keras.optimizers import SGD
from keras.utils import np_utils
import numpy as np
from scipy.misc import imresize
import argparse
import cv2
import time


scale = 0.25
epochs = 20
timestepsize = 10 # only get every n-th frame

class SutureVideoLoader:
    def __init__(self):
        print("Creating Suture video collection index...")
        self.viddir = "./data/Suturing/video/"
        labeldir = "./data/Suturing/transcriptions/"
        
        vidfiles = os.listdir(self.viddir)
        labelfiles = os.listdir(labeldir)
        maxframes = 0
        vidind = list()
        self.framelabels = list()
        for vid in vidfiles:
            video = cv2.VideoCapture(self.viddir+vid)
            minindex = maxframes
            maxframes += int(video.get(7)) #number of frames in this video
            vidind.append((maxframes, (vid, minindex) ))
            video.release()

            # put together label sets for frames of this video
            with open(labeldir+vid[:-13]+".txt") as f:
                t = f.read()
            lines = t.split(" \n")[:-1]
            vidlabelinfos = list(map(lambda x: x.split(' '), lines))
            vidlabels = [0]*(int(vidlabelinfos[0][0])-1)
            # 0 being the label of no gesture
            for labelinfo in vidlabelinfos:
                vidlabels += [int(labelinfo[2][1:])]*(int(labelinfo[1])-int(labelinfo[0])+1)
            vidlabels += [0]*(maxframes-int(vidlabelinfos[-1][1]))
            self.framelabels += vidlabels
        self.framelabels = np.array(vidlabels)
        self.maxframe = maxframes - 1
        self.videoindex = vidind
        print("Index complete. Collection frame count: %s" % maxframes)

    def getVideoForFrame(self, frame):
        # returns a tuple with the video file name and the 0 frame number
        if frame > self.maxframe:
            print("[ERROR] Frame num too high for this collection")
            return None
        for index in self.videoindex:
            if frame < index[0]:
                return index[1]
        print("[ERROR] No video found for this frame???")
        return None

    def __call__(self, frame):
        videoinfo = self.getVideoForFrame(frame)
        if videoinfo:
            framenum = frame - videoinfo[1]
            vid = cv2.VideoCapture(self.viddir+videoinfo[0])
            vid.set(1, framenum)
            ret, theframe = vid.read()
            vid.release()
            if ret == False:
                print("[ERROR] No frame returned")
                return None
            modframe = imresize(theframe, (int(480*scale), int(640*scale)))/255.0
            return np.swapaxes(np.swapaxes(modframe, 0,2),1,2)
        print("[ERROR] No video info returned")
        return None

def main():
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-s", "--save-model", type=int, default=-1,
        help="(optional) whether or not model should be saved to disk")
    ap.add_argument("-l", "--load-model", type=int, default=-1,
        help="(optional) whether or not pre-trained model should be loaded")
    ap.add_argument("-w", "--weights", type=str,
        help="(optional) path to weights file")
    args = vars(ap.parse_args())

    # initialize the optimizer and model
    print("[INFO] compiling model... " + time.asctime())
    opt = SGD(lr=0.01)
    model = LeNet.build(int(640*scale), int(480*scale), 3, 12,
        weightsPath=args["weights"] if args["load_model"] > 0 else None)
    model.compile(loss="categorical_crossentropy", optimizer=opt,
        metrics=["accuracy"])

    # load dataset
    video_loader = SutureVideoLoader()
    frame_pattern = range(0, video_loader.maxframe+1, timestepsize)
    labels = video_loader.framelabels[frame_pattern]
    ic = ImageCollection(frame_pattern, conserve_memory=True, load_func=video_loader)
    print("Number of frames considered: %s" % len(ic))
     
    # construct the training and testing splits
    print("Creating train and test sets... This will take a while... " + time.asctime())
    (trainData, testData, trainLabels, testLabels) = train_test_split(
        ic, labels.astype("int"), test_size=0.33)
    print("DONE! " + time.asctime())

    trainLabels = np_utils.to_categorical(trainLabels, 12)
    '''
    try:
        print("Len TL: %s" % len(trainLabels))
        print(trainLabels.shape)
    except:
        pass
    #'''

    testLabels = np_utils.to_categorical(testLabels, 12)

    trainData = [np.array(trainData)]
    testData = [np.array(testData)]
    '''
    try:
        print("Len TD: %s" % len(trainData))
        print(trainData.shape)
    except:
        pass
    try:
        print("Len TD[0]: %s" % len(trainData[0]))
        print(trainData[0].shape)
    except:
        pass
    try:
        print("Len TD[0][0]: %s" % len(trainData[0][0]))
        print(trainData[0][0].shape)
    except:
        pass
    try:
        print("Len TD[0][0][0]: %s" % len(trainData[0][0][0]))
    except:
        pass
    
    #'''

    # only train and evaluate the model if we are NOT loading a
    # pre-existing model
    if args["load_model"] < 0:
        print("[INFO] training... " + time.asctime())
        model.fit(trainData, trainLabels, batch_size=32, nb_epoch=epochs, verbose=1)
     
    # show the accuracy on the testing set
    print("[INFO] evaluating... " + time.asctime())
    (loss, accuracy) = model.evaluate(testData, testLabels,
        batch_size=32, verbose=1)
    print("[INFO] accuracy: {:.2f}%".format(accuracy * 100))

    print("[INFO] dumping weights to file... " + time.asctime())
    model.save_weights(args["weights"] or str(int(time.time()))+".weights.txt", overwrite=True)
    #'''

if __name__ == "__main__":
    main()