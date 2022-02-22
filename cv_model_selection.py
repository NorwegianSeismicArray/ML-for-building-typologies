import os
import sys
from pathlib import Path
from shutil import rmtree
from tempfile import TemporaryDirectory
from csv import DictWriter
import numpy as np
from numpy.lib.function_base import average
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)
from sklearn.model_selection import RepeatedKFold
import matplotlib.pyplot as plt
from typing import List



TYPOLOGIES = ['CR', 'MUR', 'S', 'SRC', 'T', 'other']


def create_symlinks(target_dir: Path, train_files: List[Path], test_files: List[Path]):
    """
    Given two lists of files, create symlinks in the target directory
    """
    if target_dir.exists():
        print('Removing', target_dir)
        
        # def recursive_unlink(d: Path):
        #     for target in d.iterdir():
        #         if target.is_symlink():
        #             Path.unlink(target)
        #         else:
        #             recursive_unlink(target)
        
        # recursive_unlink(target_dir)
        rmtree(target_dir)
    
    print('Creating', target_dir/'train')
    print('Creating', target_dir/'test')

    Path.mkdir(target_dir)
    Path.mkdir(target_dir / 'train')
    Path.mkdir(target_dir / 'test')
    for t in TYPOLOGIES:
        Path.mkdir(target_dir / 'train' / t)
        Path.mkdir(target_dir / 'test' / t)


    def make_link(f, category):
        
        f = Path(f)
        f = Path.resolve(f)
        cat = f.parent.__str__().split('/')[-1]
        link = Path(Path.resolve(target_dir) / category / cat / f.name)
        try:
            link.symlink_to(f)
            return 0
        except FileExistsError:
            return 1

    n_duplicates = 0

    for f in train_files:
        n_duplicates += make_link(f, 'train')

    for f in test_files:
        n_duplicates += make_link(f, 'test')

    print(f'Skipped {n_duplicates} duplicate images')
    
    return target_dir / 'train', target_dir / 'test'


def get_all_files(source_dir: Path):

    def get_files_in_dir(d):
        files = []
        for f in d.iterdir():
            if f.is_file():
                files.append(f)
            elif f.is_dir():
                files += get_files_in_dir(f)
        return files
    
    allfiles = get_files_in_dir(source_dir)

    return allfiles


def get_files_from_list(filename):

    with open(filename) as fin:
        files = [l.strip() for l in fin.readlines()]

    return files


def train_base_model(
        train_ds,
        val_ds,
        image_size,
        num_classes,
        pretrained_model_fn,
        preprocessing_fn,
        num_epochs
):
    """
    pretrained_model_fn: function from tf.keras.applications, e.g.
    tf.keras.applications.Xception()
    """

    print('Training {} with {} as preproecssing'.format(
        pretrained_model_fn.__str__(),
        preprocessing_fn.__str__()
    ))

    imgshape = image_size + (3, )

    label_names = sorted(['CR', 'MUR', 'other', 'S', 'SRC', 'T'])
    label_names = {i: l for i, l in enumerate(label_names)}

    data_augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal'),
            tf.keras.layers.experimental.preprocessing.RandomRotation(0.025),
            tf.keras.layers.experimental.preprocessing.RandomZoom(0.2),
            #tf.keras.layers.experimental.preprocessing.RandomContrast(0.1),
        ],
        name='data_augmentation'
    )

    inputs = tf.keras.layers.Input(shape=imgshape)
    x = data_augmentation(inputs)
    x = preprocessing_fn(x)

    base_model = pretrained_model_fn(
        weights='imagenet',
        include_top=False,
        input_shape=imgshape
    )
    base_model.trainable = False

    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.20)(x)
    outputs = tf.keras.layers.Dense(units=num_classes, activation='softmax')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.CategoricalCrossentropy(),    #label_smoothing=0.05),
        metrics=[
            'categorical_accuracy',
            tf.keras.metrics.Precision(),
            tf.keras.metrics.Recall(),
            tf.keras.metrics.AUC()
        ]
    )

    earlystop = tf.keras.callbacks.EarlyStopping(patience=10, verbose=1, restore_best_weights=True)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5, verbose=1)

    # 1: Train with base model fixed
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=num_epochs,
        callbacks=[earlystop, reduce_lr],
        verbose=2
    )

    return model, history


def get_model_spec(num: int):

    models = [
        {
            'name': 'Xception',
            'model_fn': tf.keras.applications.Xception,
            'preproc_fn': tf.keras.applications.xception.preprocess_input,
            'imgs_size': (299, 299)
        },
        {
            'name': 'VGG16',
            'model_fn': tf.keras.applications.VGG16,
            'preproc_fn': tf.keras.applications.vgg16.preprocess_input,
            'imgs_size': (224, 224)
        },
        {
            'name': 'VGG19',
            'model_fn': tf.keras.applications.VGG19,
            'preproc_fn': tf.keras.applications.vgg19.preprocess_input,
            'imgs_size': (224, 224)
        },
        # {
        #     'name': 'ResNet50',
        #     'model_fn': tf.keras.applications.ResNet50,
        #     'preproc_fn': tf.keras.applications.resnet.preprocess_input,
        #     'imgs_size': (224, 224)
        # },
        # {
        #     'name': 'ResNet101',
        #     'model_fn': tf.keras.applications.ResNet101,
        #     'preproc_fn': tf.keras.applications.resnet.preprocess_input,
        #     'imgs_size': (224, 224)
        # },
        # {
        #     'name': 'ResNet151',
        #     'model_fn': tf.keras.applications.ResNet152,
        #     'preproc_fn': tf.keras.applications.resnet.preprocess_input,
        #     'imgs_size': (224, 224)
        # },
        {
            'name': 'ResNet50V2',
            'model_fn': tf.keras.applications.ResNet50V2,
            'preproc_fn': tf.keras.applications.resnet_v2.preprocess_input,
            'imgs_size': (224, 224)
        },
        {
            'name': 'ResNet101V2',
            'model_fn': tf.keras.applications.ResNet101V2,
            'preproc_fn': tf.keras.applications.resnet_v2.preprocess_input,
            'imgs_size': (224, 224)
        },
        {
            'name': 'ResNet151V2',
            'model_fn': tf.keras.applications.ResNet152V2,
            'preproc_fn': tf.keras.applications.resnet_v2.preprocess_input,
            'imgs_size': (224, 224)
        },
        {
            'name': 'InceptionV3',
            'model_fn': tf.keras.applications.InceptionV3,
            'preproc_fn': tf.keras.applications.inception_v3.preprocess_input,
            'imgs_size': (299, 299)
        },
        {
            'name': 'InceptionResNetV2',
            'model_fn': tf.keras.applications.InceptionResNetV2,
            'preproc_fn': tf.keras.applications.inception_resnet_v2.preprocess_input,
            'imgs_size': (299, 299)
        },
        {
            'name': 'DenseNet121',
            'model_fn': tf.keras.applications.DenseNet121,
            'preproc_fn': tf.keras.applications.densenet.preprocess_input,
            'imgs_size': (224, 224)
        },
        {
            'name': 'DenseNet169',
            'model_fn': tf.keras.applications.DenseNet169,
            'preproc_fn': tf.keras.applications.densenet.preprocess_input,
            'imgs_size': (224, 224)
        },
        {
            'name': 'DenseNet201',
            'model_fn': tf.keras.applications.DenseNet201,
            'preproc_fn': tf.keras.applications.densenet.preprocess_input,
            'imgs_size': (224, 224)
        },

    ]

    return models[num]



if __name__ == '__main__':

    modelnum = int(sys.argv[1])
    print(f'Running model number {modelnum}')

    files = np.array(get_files_from_list('imgs_for_training.txt'))

    metrics = [
        'epochs',
        'train_loss',
        'val_loss',
        'accuracy',
        'precision_macro',
        'precision_weighted',
        'recall_macro',
        'recall_weighted',
        'f1_macro',
        'f1_weighted',
        'roc_macro',
        'roc_weighted'
    ]
    
    model_spec = get_model_spec(modelnum)
    print('Model name:', model_spec['name'])

    with TemporaryDirectory() as tmpdir:

        results_file_name = 'cv_results_{}_job_{}.txt'.format(model_spec['name'], modelnum)
        results_file = open(results_file_name, 'w', newline='')
        results_csv = None
        

        rkf = RepeatedKFold(n_splits=4, n_repeats=3)

        for train_idxs, test_idxs in rkf.split(files):
            
            trainfiles = files[train_idxs]
            testfiles = files[test_idxs]

            traindir, testdir = create_symlinks(
                Path(tmpdir) / 'cross_validation',
                trainfiles,
                testfiles
            )

            imgsize = model_spec['imgs_size']
            batchsize = 32
            
            print('Preparing train dataset:')
            train_ds = tf.keras.preprocessing.image_dataset_from_directory(
                traindir,
                image_size=imgsize,
                batch_size=batchsize,
                label_mode='categorical'
            )

            print('Preparing val dataset:')
            val_ds = tf.keras.preprocessing.image_dataset_from_directory(
                testdir,
                image_size=imgsize,
                batch_size=batchsize,
                label_mode='categorical'
            )

            model, hist = train_base_model(
                train_ds, 
                val_ds, 
                imgsize,
                num_classes=len(TYPOLOGIES),
                pretrained_model_fn=model_spec['model_fn'],
                preprocessing_fn=model_spec['preproc_fn'],
                num_epochs=100
            )
            
            completed_epochs = len(hist.history['loss'])

            #val_preds = model.predict(val_ds)
            #val_truth = np.vstack([y for x, y in val_ds])
            #val_truth = val_truth.reshape(val_preds.shape)

            val_preds = []
            val_truth= []
            for bx, by in val_ds:   # load batches
                bpreds = model.predict(bx)
                for p, y in zip(bpreds, by):
                    val_truth.append(y)
                    val_preds.append(p)

            val_preds_vec = np.argmax(val_preds, axis=1)
            val_truth_vec = np.argmax(val_truth, axis=1)
            
            print('val_preds_vec', val_preds_vec[:30])
            print('val_truth_vec', val_truth_vec[:30])

            # Compute and report metrics
            train_loss = model.evaluate(train_ds, verbose=0)[0]
            val_loss = model.evaluate(val_ds, verbose=0)[0]
            
            print('train_loss', train_loss)
            print('val_loss', val_loss)
            
            results = {
                'epochs': completed_epochs,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'accuracy_skl': accuracy_score(val_truth_vec, val_preds_vec),
                'precision_skl_macro': precision_score(val_truth_vec, val_preds_vec, average='macro'),
                'precision_skl_wgt': precision_score(val_truth_vec, val_preds_vec, average='weighted'),
                'recall_skl_macro': recall_score(val_truth_vec, val_preds_vec, average='macro'),
                'recall_skl_wgt': recall_score(val_truth_vec, val_preds_vec, average='weighted'),
                'f1_skl_macro': f1_score(val_truth_vec, val_preds_vec, average='macro'),
                'f1_skl_wgt': f1_score(val_truth_vec, val_preds_vec, average='weighted'),
                'auc_skl_macro': roc_auc_score(val_truth, val_preds, average='macro', multi_class='ovr'),
                'auc_skl_wgt': roc_auc_score(val_truth, val_preds, average='weighted', multi_class='ovr')
            }

            # Compute Keras/TF metrics
            
            m = tf.keras.metrics.CategoricalAccuracy()
            m.update_state(val_truth, val_preds)
            results['accuracy_tf'] = m.result().numpy()

            m = tf.keras.metrics.Precision()
            m.update_state(val_truth, val_preds)
            results['precision_tf'] = m.result().numpy()

            m = tf.keras.metrics.Recall()
            m.update_state(val_truth, val_preds)
            results['recall_tf'] = m.result().numpy()

            m = tf.keras.metrics.AUC()
            m.update_state(val_truth, val_preds)
            results['auc_tf'] = m.result().numpy()


            if results_csv is None:
                results_csv = DictWriter(results_file, fieldnames=results.keys(), delimiter=';')
                results_csv.writeheader()
            
            results_csv.writerow(results)

            for k, v in results.items():
                print(k, v)


        results_file.close()


