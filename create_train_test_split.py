import os
from pathlib import Path
from shutil import rmtree
from random import uniform


TYPOLOGIES = ['CR', 'MUR', 'S', 'SRC', 'T', 'other']


def create_train_test_symlinks(source_dir, test_fraction, overwrite=False):


    assert 0.0 < test_fraction < 1.0

    # Randomly choose test/train
    def rnd_out_dir():
        d = 'test' if uniform(0, 1) < test_fraction else 'train'
        return d 

    source_dir = Path(source_dir)

    # Recurse directories to find expected subdirectories
    def find_source_dirs(top_dir):
        dirs = []
        for d in top_dir.iterdir():
            if d.name in TYPOLOGIES:
                dirs.append(d)
            elif d.is_dir():
                dirs += find_source_dirs(d)
            else:
                pass
        return dirs


    trainfile = 'imgs_for_training.txt'
    testfile = 'imgs_for_testing.txt'

    if not overwrite:
        if Path(trainfile).exists() or Path(testfile).exists():
            raise RuntimeError(f'Files already exist: {trainfile} {testfile}')

    f_train_set = open(trainfile, 'w')
    f_test_set = open(testfile, 'w')

    source_dirs = find_source_dirs(source_dir)
    for sd in source_dirs:
        for img_file in sd.iterdir():

            if uniform(0, 1) < test_fraction:
                f_test_set.write(img_file.__str__() + '\n')
            else:
                f_train_set.write(img_file.__str__() + '\n')

    f_train_set.close()
    f_test_set.close()



if __name__ == '__main__':

    create_train_test_symlinks('cleaned_images_for_ML_final', 0.2)
