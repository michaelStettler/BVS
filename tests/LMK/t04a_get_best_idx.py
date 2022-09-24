import os
import numpy as np

from utils.load_config import load_config

"""
run: python -m tests.LMK.t04a_get_best_idx
"""

if __name__ == '__main__':

    # define configuration
    config_path = 'LMK_t04_run_NRE_on_LMK_data_m0001.json'
    # load config
    config = load_config(config_path, path='configs/LMK')
    print("-- Config loaded --")
    print()

    # declare variables
    avatar_names = ['jules', 'malcolm', 'ray', 'aia', 'bonnie', 'mery']
    avatar_name = avatar_names[1]
    print("avatar_name:", avatar_name)

    best_idx = np.load(os.path.join(config['directory'], 'best_indexes', avatar_name + '_best_idx' + '.npy'))
    print("best_idx:", best_idx)

    """
    Malcolm (from 83.89 to 99.16): 
        - 9         malcolm_joy_63.png      -> 96
        - 1693      malcolm_anger_753.png   -> 12546
        - 1040      malcolm_sadness_631.png -> 7829
        - 203       malcolm_surprise_249.png-> 1591
        - 354       malcolm_fear_277.png    -> 2674
        - 4620      malcolm_disgust_1027.png-> 34136
        
    Ray (from 81.15 to 97.04): 
        - 2165      ray_joy_505.png         -> 14172
        - 1914      ray_anger_682.png       -> 12476
        - 5753      ray_sadness_176.png     -> 37443
        - 529       ray_surprise_106.png    -> 40043
        - 5455      ray_fear_350.png        -> 35384
        - 1586      ray_disgust_1149.png    -> 10239
        
    Aia (from 88.29 to 98.96): 
        - 1585      aia_joy_830.png         -> 8826
        - 5023      aia_anger_473.png       -> 27756
        - 573       aia_sadness_485.png     -> 3154
        - 3414      aia_surprise_683.png    -> 18952
        - 3544      aia_fear_817.png        -> 19615
        - 4273      aia_disgust_762.png     -> 23813
        
    Bonnie (from 77.63 to 99.24)
        - 1306      
        - 95      
        - 4531       
        - 981      
        - 3763      
        - 2471      
        
    Mery (from 72.68 to 89.41):
        - 1462:     mery_joy_784.png        ->  11071
        - 5543:     mery_anger_1123.png     ->  41271
        - 3721:     mery_sadness_338.png    ->  27809
        - 69:       mery_surprise_925.png   ->  691
        - 1669:     mery_fear_696.png       ->  12415
        - 4853:     mery_disgust_133.png    ->  36013
        
    """
