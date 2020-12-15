import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd


# settings for creating dataset
# also create pictures, if false only csv file is generated
create_pictures = True
csv_path = "/app/Data/Dataset/ExpressionMorphing/ExpressionMorphing.csv"
load_path_base = "/app/Data/Dataset/stimuli"
save_folder = "/app/Data/Dataset/ExpressionMorphing/images"

#constants
anger_list = [0.0,0.25,0.5,0.75, 1.0]
human_expression_list = [0.0, 0.25, 0.5, 0.75, 1.0]
avatar_human_list = [True, False]
num_image = 150
dataframe = pd.DataFrame(columns = ['image_path', 'image_name', 'category', 'monkey_avatar', 'anger', 'fear',
                                    'monkey_expression', 'human_expression', 'neutral_expression'])


for avatar_human in avatar_human_list:
    for anger in anger_list:
        for human_expression in human_expression_list:
            # load picture into numpy
            if avatar_human:
                load_folder = os.path.join(load_path_base, "humanAvatar")
            else:
                load_folder = os.path.join(load_path_base, "monkeyAvatar")
            load_folder = os.path.join(load_folder, "Normal")
            folder_name = ""
            if avatar_human:
                folder_name = folder_name + "HumanAvatar"
            else:
                folder_name = folder_name + "MonkeyAvatar"
            folder_name = folder_name + "_Anger_{}_Fear_{}_Monkey_{}_Human_{}"
            folder_name = folder_name.format(anger, 1 - anger, 1 - human_expression, human_expression)

            for i_image in range(num_image):
                print("[LOOP]", folder_name)

                save_image_name = folder_name + (".%04d.png" % i_image)
                save_path = os.path.join(save_folder, save_image_name)

                if create_pictures:
                    load_image_name = folder_name + (".%04d.jpg" % i_image)
                    load_path = os.path.join(load_folder, folder_name, load_image_name)
                    if not os.path.exists(load_path):
                        load_image_name = folder_name + (".%04d.jpeg" % i_image)
                        load_path = os.path.join(load_folder, folder_name, load_image_name)
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

                # judgement of picture
                # TODO change from manual to deformation data
                category = 0
                if (i_image >=35) and (i_image <=80):
                    if anger >= 0.5:
                        category = 2
                    else:
                        category = 1
                # category = 0
                # if avatar_human:
                #     if anger == 0.0:
                #         if (i_image >= 36) and (i_image <= 83):
                #             category = 1
                #     else:
                #         if (i_image >= 37) and (i_image <= 83):
                #             category = 2
                # else:
                #     if anger == 0.0:
                #         if (i_image >= 37) and (i_image <= 80):
                #             category = 1
                #     else:
                #         if (i_image >= 35) and (i_image <= 76):
                #             category = 2
                # columns = ['image_path', 'image_name', 'category', 'monkey_avatar', 'anger', 'fear',
                #                                     'monkey_expression', 'human_expression', 'neutral_expression']
                dataframe.loc[len(dataframe)] = ([save_path, save_image_name, category, not avatar_human, anger, 1 - anger, 1 - human_expression, human_expression, 0.0])
                print("Picture", len(dataframe), "of 7500")

# write csv
dataframe.to_csv(csv_path)