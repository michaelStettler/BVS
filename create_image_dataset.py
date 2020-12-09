import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd



anger_list = [0.0, 1.0]
avatar_human_list = [True, False]
num_image = 150
dataframe = pd.DataFrame(columns = ['image_path', 'image_name', 'category', 'monkey_avatar', 'anger', 'fear',
                                    'monkey_expression', 'human_expression', 'neutral_expression'])
csv_path = "/app/Data/Dataset/ArtificialFaces/ArtificialFaces_600.csv"


for anger in anger_list:
    for avatar_human in avatar_human_list:
        # load picture into numpy
        load_folder = "/app/Data/Dataset/stimuli"
        save_folder = "/app/Data/Dataset/ArtificialFaces/ArtificialFaces_600"
        if avatar_human:
            load_folder = os.path.join(load_folder, "humanAvatar")
        else:
            load_folder = os.path.join(load_folder, "monkeyAvatar")
        load_folder = os.path.join(load_folder, "Normal")
        folder_name = ""
        if avatar_human:
            folder_name = folder_name + "HumanAvatar"
        else:
            folder_name = folder_name + "MonkeyAvatar"
        folder_name = folder_name + "_Anger_%.1f_Fear_%.1f_Monkey_0.0_Human_1.0"
        folder_name = folder_name % (anger, 1 - anger)

        for i_image in range(num_image):
            print("[LOOP] picture {} of {} of anger={} and avatar_human={}".format(i_image, num_image, anger, avatar_human))

            load_path = os.path.join(load_folder, folder_name, folder_name + (".%04d.jpeg" % i_image))
            save_image_name = folder_name + (".%04d.png" % i_image)
            save_path = os.path.join(save_folder, save_image_name)
            image = cv2.imread(load_path)
            #print("[LOAD] image shape", image.shape)

            # cut monkey picture from (720,1280) to (720,720)
            if not avatar_human:
                image = image[:,280:1000]
            #print("[CUT] image shape", image.shape)

            # resize to (224,224,3)
            image = cv2.resize(image, (224,224))
            #print("[RESIZE] image shape", image.shape)

            # save image
            cv2.imwrite(save_path, image)

            # manual judgement of picture
            category = 0
            if avatar_human:
                if anger == 0.0:
                    if (i_image >= 36) and (i_image <= 83):
                        category = 1
                else:
                    if (i_image >= 37) and (i_image <= 83):
                        category = 2
            else:
                if anger == 0.0:
                    if (i_image >= 37) and (i_image <= 80):
                        category = 1
                else:
                    if (i_image >= 35) and (i_image <= 76):
                        category = 2

            dataframe.loc[len(dataframe)] = ([save_path, save_image_name, category, not avatar_human, anger+0, 1-anger, 0.0, 1.0, 0.0 ])
            print("len(dataframe)", len(dataframe))

# write csv
dataframe.to_csv(csv_path)