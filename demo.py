import os, sys
import csv
from itertools import chain
from sklearn import metrics
from datasets.config import config as datasets_config
from datasets.base import get_generators
# from models.config import config as models_config
from keras.models import load_model
from processors.process_image import process_image
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import argparse
from natsort import natsorted, ns
import cv2
import keras
import tensorflow as tf
os.environ["CUDA_VISIBLE_DEVICES"]="1"

# construct the argument parse and parse the arguments

# Put thorough model predictor
print("python:{}, keras:{}, tensorflow: {}".format(sys.version, keras.__version__, tf.__version__))

#print("Predicting now...")
#predicted = model.predict(X_test)
#rmse = np.sqrt(((predicted - y_test) ** 2).mean(axis=0))

# Replace with dynamic class names.
CLASSES =[ "ApplyEyeMakeup",
"Archery",
"BalanceBeam",
"BaseballPitch",
"BasketballDunk",
"BenchPress",
"Biking",
"Billiards",
"BlowDryHair",
"BlowingCandles",
"BodyWeightSquats",
"Bowling",
"BoxingPunchingBag",
"BoxingSpeedBag",
"BreastStroke",
"BrushingTeeth",
"CleanAndJerk",
"CliffDiving",
"CricketBowling",
"CricketShot",
"CuttingInKitchen",
"Diving",
"Drumming",
"Fencing",
"FieldHockeyPenalty",
"FloorGymnastics",
"FrisbeeCatch",
"FrontCrawl",
"GolfSwing",
"Haircut",
"Hammering",
"HammerThrow",
"HandstandPushups",
"HandstandWalking",
"HeadMassage",
"HighJump",
"HorseRace",
"HorseRiding",
"HulaHoop",
"IceDancing",
"JavelinThrow",
"JugglingBalls",
"JumpingJack",
"JumpRope",
"Kayaking",
"Knitting",
"LongJump",
"Lunges",
"MilitaryParade",
"Mixing",
"MoppingFloor",
"Nunchucks",
"ParallelBars",
"PizzaTossing",
"PlayingCello",
"PlayingDaf",
"PlayingDhol",
"PlayingFlute",
"PlayingGuitar",
"PlayingPiano",
"PlayingSitar",
"PlayingTabla",
"PlayingViolin",
"PoleVault",
"PommelHorse",
"PullUps",
"Punch",
"PushUps",
"Rafting",
"RockClimbingIndoor",
"RopeClimbing",
"Rowing",
"SalsaSpin",
"ShavingBeard",
"Shotput",
"SkateBoarding",
"Skiing",
"Skijet",
"SkyDiving",
"SoccerJuggling",
"SoccerPenalty",
"StillRings",
"SumoWrestling",
"Surfing",
"Swing",
"TableTennisShot",
"TaiChi",
"TennisSwing",
"ThrowDiscus",
"TrampolineJumping",
"Typing",
"UnevenBars",
"VolleyballSpiking",
"WalkingWithDog",
"WallPushups",
"WritingOnBoard",
"YoYo",
"ExitDoor",
"WalkingPast"]


def build_image_sequence(frames):
    """Given a set of frames (filenames), build our sequence."""
    return [process_image(x, (224, 224, 3)) for x in frames]

# Load a video sequence
def load_images(sequence_path, sequence_number):
    images = []
    imageArray = []

    for _ in range(50):
        X = []
        frames = os.listdir(os.path.join(sequence_path))
        frames = natsorted(frames, key=lambda y: y.lower())
        sequence = build_image_sequence(frames)
        X.append(sequence)

    convertArray = np.array([X])

    print('frames-shape: ', convertArray.shape)
    return convertArray

    # frames = get_frames_for_sample(sample)
    # frames = rescale_list(frames, seq_length)



    # Build the image sequence

    # for f in os.listdir(os.path.join(sequence_path)):
    #     if f.startswith(str(sequence_number)):
    #         images.append(f)

    # images = natsorted(images, key=lambda y: y.lower())

    # Convert images to Numpy Array (similar)
    # for i in images:
    #     im = cv2.imread(os.path.join(sequence_path, i))
    #     imageArray.append(im)

    # Determine the width and height from the first image
    # try:
    #     frame = cv2.imread(images[0])
    #     height, width, channels = frame.shape
    # except:
    #     height = 240
    #     width = 320
    #     channels = 3




def evaluate(model, images_array):
    # Get a subset of the test data. We do this so we can compare the predictions
    # to the actual, without having to deal with the randomization of the generators.
    # TODO There has to be a better way to do this part
    # eval_data = list(x for x in images_array)
    # get each sample from each step and batch
    x = []
    y = []
    print('Attempting images_array...')
    print('[INFO]: Images Array: ', images_array.shape)

    # np.savetxt("file_name.csv", list(x for x in images_array), delimiter=",", fmt='%s')
    # print('[INFO]: Saved file')

    for row in images_array:
        for sample in row[0][0]:
            x.append(sample)
        # for sample in row[1]:
        #     y.append(sample)
    eval_x = np.array(x)
    # eval_y = np.array(y)
    print(6, eval_x.shape)
    # print(7, eval_y.shape)


    # Now predict with the trained model and compare to the actual.
    print('[INFO]: Evaluating...')
    predictions = model.predict(eval_x, verbose=1, batch_size=8)
    print('[INFO]: Predictions...', predictions)
    predictions_index = predictions.argmax(axis=-1)
    # actual_index = eval_y.argmax(axis=-1)
    # confusion_matrix = metrics.confusion_matrix(actual_index, predictions_index,
    #                                             labels=list(range(len(CLASSES))))
    # print(9, confusion_matrix)
    print('Predictions index: ', predictions_index)

    # Visualize the confusion matrix.
    # df_cm = pd.DataFrame(confusion_matrix, index=[i for i in CLASSES],
                         # columns=[i for i in CLASSES])
    # plt.figure(figsize=(10, 7))
    # sns.heatmap(df_cm, annot=True, fmt='g')
    # plt.ylabel('True label')
    # plt.xlabel('Predicted label')
    # plt.show()

def main(model_name, dataset, model_path, images_array):
    paths = models_config['models'][model_name]['paths']
    input_shapes = [x['input_shape'] for x in paths]
    preprocessing_steps = [x['preprocessing'] for x in paths]
    nb_classes = datasets_config['datasets'][dataset]['nb_classes']

    # Get the model.
    model = load_model('../../Desktop/lstm-features.026-1.069.hdf5')
    print('[INFO]: Model got loaded')

    # Get the data generators.
    #generators = get_generators(dataset, 50, nb_classes, input_shapes, preprocessing_steps, 32)
    #print(7)
    #__, val_gen = generators

    evaluate(model, images_array)

if __name__ == '__main__':
    models_config = {
        'models': {
            'c3d': {
                'paths': [
                    {
                        'preprocessing': 'images',
                        'input_shape': (80, 80, 3),
                    },
                ],
            },
            'vgg_rnn': {
                'paths': [
                    {
                        'preprocessing': 'images',
                        'input_shape': (80, 80, 3),
                    },
                ],
            },
            'conv3d': {
                'paths': [
                    {
                        'preprocessing': 'images',
                        'input_shape': (80, 80, 3),
                    },
                ],
            },
            'lrcn': {
                'paths': [
                    {
                        'preprocessing': 'images',
                        'input_shape': (80, 80, 3),
                    },
                ],
            },
        },
    }
    ap = argparse.ArgumentParser()
    ap.add_argument("-mn", "--model_name", action='store', type=str, default="test", required=True,
    	help="The model to be trained")
    ap.add_argument("-ds", "--dataset", action="store",
    	help="The dataset to train on")
    ap.add_argument("-p", "--model_path", action="store",
                        help="Saved model to load.")
    ap.add_argument("-sp", "--sequence_path", type=str,
                        help="Sequence path")
    ap.add_argument("-sn", "--sequence_number", type=str,
                        help="Sequence 4 digit number")
    args = vars(ap.parse_args())

    images_array = load_images(args['sequence_path'], args['sequence_number'])

    main(args['model_name'], args['dataset'], args['model_path'], images_array)



# Load the web server from the cam IP address

# Grab images at 10 FPS

# Convert those images to a Numpy Array  in batches of 25 and add to a queue

# Ad sequential batches and pass through model predictor

# Put images through model predictor in a batch
