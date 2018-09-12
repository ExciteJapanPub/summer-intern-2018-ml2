# coding: utf-8
import os
import glob
import pandas as pd
import subprocess

DATADIR = "/Users/excite3/Work/summer-intern-2018-ml2/FISH_data/raw"
IMG_OUTDIR = os.path.join(DATADIR, "images")
if not os.path.exists(IMG_OUTDIR):
    os.mkdir(IMG_OUTDIR)

TRAIN_IMG_OUTDIR = os.path.join(IMG_OUTDIR, "train")
if not os.path.exists(TRAIN_IMG_OUTDIR):
    os.mkdir(TRAIN_IMG_OUTDIR)

TEST_IMG_OUTDIR = os.path.join(IMG_OUTDIR, "test")
if not os.path.exists(TEST_IMG_OUTDIR):
    os.mkdir(TEST_IMG_OUTDIR)

CSV_OUTDIR = os.path.join(DATADIR, "labels")
if not os.path.exists(CSV_OUTDIR):
    os.mkdir(CSV_OUTDIR)

TRAIN_CSV = os.path.join(CSV_OUTDIR, "train.csv")

TEST_CSV = os.path.join(CSV_OUTDIR, "test.csv")

TEST_NUM = 5

FISH_LABEL_LIST = [['ittennhuedai', '0'], ['haokoze', '1'], ['gonnzui', '2'],
['soushihagi', '3'], ['gigi', '4'], ['aigo', '5'], ['other', '6']]


def rename():
    train_df = pd.DataFrame([])
    test_df = pd.DataFrame([])
    for i in range(len(FISH_LABEL_LIST)):
        fish_name = FISH_LABEL_LIST[i][0]
        label = FISH_LABEL_LIST[i][1]
        img_dir = os.path.join(DATADIR, fish_name)
        fpath_list = glob.glob(os.path.join(img_dir, "*"))
        print(f"fpath_list¥n{fpath_list}")
        for i, fpath in enumerate(fpath_list):
            id_s = fpath.rfind("/")
            fname = fpath[id_s+1:]
            out_name = label + "_" + str(i).zfill(3) + ".jpg"
            # print(f"only name{fname}")
            if i < len(fpath_list)-TEST_NUM:
                # train data
                train_df = pd.concat([train_df, pd.DataFrame([out_name, label])], axis=1)
                cmd = "cp '" + fpath + "' " + os.path.join(TRAIN_IMG_OUTDIR, out_name)
                cmds = ["cp", fpath, os.path.join(TRAIN_IMG_OUTDIR, out_name)]
            else:
                # test data
                test_df = pd.concat([test_df, pd.DataFrame([out_name, label])], axis=1)
                cmd = "cp '" + fpath + "' " + os.path.join(TEST_IMG_OUTDIR, out_name)
                cmds = ["cp", fpath, os.path.join(TEST_IMG_OUTDIR, out_name)]
            # run cp command
            print(f"$ {cmd}")
            subprocess.check_call(cmds)
    # output csv
    print("output csv...")
    train_df.T.to_csv(TRAIN_CSV, index=False, header=False)
    test_df.T.to_csv(TEST_CSV, index=False, header=False)

    return 0


if __name__ == '__main__':
    rename()




