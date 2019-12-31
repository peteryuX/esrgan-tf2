"""Rename DIV2K data downlowed from https://data.vision.ee.ethz.ch/cvl/DIV2K/.
   (This preprocessing code is copied and modified from official implement:
    https://github.com/open-mmlab/mmsr/tree/master/codes/data_scripts)"""
import os
import glob


def main():
    folder = './data/DIV2K/DIV2K_train_LR_bicubic/X4'
    DIV2K(folder)
    print('Finished.')


def DIV2K(path):
    img_path_l = glob.glob(os.path.join(path, '*'))
    for img_path in img_path_l:
        new_path = img_path.replace('x2', '').replace('x3', '').replace(
            'x4', '').replace('x8', '')
        os.rename(img_path, new_path)


if __name__ == "__main__":
    main()