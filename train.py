from pathlib import Path
from argparse import ArgumentParser
from shutil import rmtree
from typing import List
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, f1_score, roc_auc_score
import matplotlib.pyplot as plt


TYPOLOGIES = ['CR', 'MUR', 'S', 'SRC', 'T', 'other']



def train(train_dataset, val_dataset, image_shape, num_classes, save_as):

    imgsize = image_shape[:-1]

    label_names = sorted(TYPOLOGIES)
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

   
    # Plot
    """
    plt.figure(figsize=(10, 10))
    for images, labels in train_ds.take(1):
        for i in range(9):
            augmented_images = data_augmentation(images)
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(augmented_images[0].numpy().astype("uint8"))
            ax.set_yticklabels([])
            ax.set_xticklabels([])
            lbl = label_names[(np.argmax(labels[0], axis=0))]
            plt.xlabel(lbl)
            print(labels[0])
        plt.show()
    """


    inputs = tf.keras.layers.Input(shape=image_shape)
    x = data_augmentation(inputs)
    x = tf.keras.applications.densenet.preprocess_input(x)
    #x = tf.keras.applications.resnet_v2.preprocess_input(x)
    #x = tf.keras.applications.xception.preprocess_input(x)

    cnn_model = tf.keras.applications.DenseNet201(
    #cnn_model = tf.keras.applications.ResNet50V2(
    #cnn_model = tf.keras.applications.Xception(
        weights='imagenet',
        include_top=False,
        input_shape=imgsize + (3,)
    )
    cnn_model.trainable = False
    
    x = cnn_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    #x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.20)(x)
    #x = tf.keras.layers.Dense(128, activation='relu')(x)
    #x = tf.keras.layers.Dropout(0.25)(x)
    outputs = tf.keras.layers.Dense(units=num_classes, activation='softmax')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    optimizer = tf.keras.optimizers.Adam(lr=0.001)
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.CategoricalCrossentropy(),#label_smoothing=0.1),
        metrics=['categorical_accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )
    print(model.summary())


    earlystop = tf.keras.callbacks.EarlyStopping(patience=10, verbose=1, restore_best_weights=True)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5, verbose=1)

    # 1: Train with ResNet model fixed
    epochs = 50
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        callbacks=[earlystop, reduce_lr],
        verbose=2
    )

    acc = history.history['categorical_accuracy']
    val_acc = history.history['val_categorical_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    pretrain_epochs = len(loss)

    model.save(save_as + '_pre_finetuning')


    # 2: Unfreeze highest block(s), train more
    unfreeze_from = 'conv5' #densenet
    #unfreeze_from = 'conv4' #resnet
    #unfreeze_from = 'block10_sepconv1'
    unfreeze_layer_num = len(cnn_model.layers)
    for i, layer in enumerate(cnn_model.layers):
        if layer.name.startswith(unfreeze_from):
            unfreeze_layer_num = i
            print(f'unfreezing from layer {i} ({layer.name})')
            break
    cnn_model.trainable = True

    for layer in cnn_model.layers[:unfreeze_layer_num]:
        layer.trainable = False
    
    # Fix batchnorm layers 
    for layer in cnn_model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
            #print('Setting layer to const:', layer.name)

    
    
    optimizer = tf.keras.optimizers.Adam(lr=0.00005)
    model.compile(
        optimizer=optimizer,
        #loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=['categorical_accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )
    print(model.summary())


    earlystop = tf.keras.callbacks.EarlyStopping(patience=5, verbose=1, restore_best_weights=True)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5, verbose=1)

    total_epochs = pretrain_epochs + 50
    history_tune = model.fit(
        train_dataset,
        validation_data=val_dataset,
        callbacks=[earlystop, reduce_lr],
        initial_epoch=pretrain_epochs,
        epochs=total_epochs,
        verbose=2
    )

    acc += history_tune.history['categorical_accuracy']
    val_acc += history_tune.history['val_categorical_accuracy']
    loss += history_tune.history['loss']
    val_loss += history_tune.history['val_loss']

    model.save(save_as)

    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.ylim([0.5, 1])
    plt.plot([pretrain_epochs-1, pretrain_epochs-1], plt.ylim(), label='Start Fine Tuning')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.ylim([0, 1.0])
    plt.plot([pretrain_epochs-1, pretrain_epochs-1], plt.ylim(), label='Start Fine Tuning')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.show()

    return model





def evaluate(model, dataset, labels, include_confusion_matrix=True):

    print(model.evaluate(dataset, verbose=0))

    preds = []
    targets= []
    for bx, by in dataset:   # load batches
        
        bpreds = model.predict(bx)
        for p, y in zip(bpreds, by):
            targets.append(y)
            preds.append(p)

    preds = np.array(preds)
    targets = np.array(targets)

    preds_manyhot = np.argmax(preds, axis=1)
    targets_manyhot = np.argmax(targets, axis=1)

    #print('accuracy:', accuracy_score(targets_manyhot, preds_manyhot))
    #print('precision (micro):', precision_score(targets_manyhot, preds_manyhot, average='micro'))
    #print('precision (macro):', precision_score(targets_manyhot, preds_manyhot, average='macro'))
    #print('precision (weighted):', precision_score(targets_manyhot, preds_manyhot, average='weighted'))
    #print('recall (micro):', recall_score(targets_manyhot, preds_manyhot, average='micro'))
    #print('recall (macro):', recall_score(targets_manyhot, preds_manyhot, average='macro'))
    #print('recall (weighted):', recall_score(targets_manyhot, preds_manyhot, average='weighted'))

    print('accuracy_skl:', accuracy_score(targets_manyhot, preds_manyhot))
    print('precision_skl_macro', precision_score(targets_manyhot, preds_manyhot, average='macro'))
    print('precision_skl_wgt', precision_score(targets_manyhot, preds_manyhot, average='weighted'))
    print('recall_skl_macro', recall_score(targets_manyhot, preds_manyhot, average='macro'))
    print('recall_skl_wgt', recall_score(targets_manyhot, preds_manyhot, average='weighted'))
    print('f1_skl_macro', f1_score(targets_manyhot, preds_manyhot, average='macro'))
    print('f1_skl_wgt', f1_score(targets_manyhot, preds_manyhot, average='weighted'))
    print('auc_skl_macro', roc_auc_score(targets, preds, average='macro', multi_class='ovr'))
    print('auc_skl_wgt', roc_auc_score(targets, preds, average='weighted', multi_class='ovr'))
    

    m = tf.keras.metrics.CategoricalAccuracy()
    m.update_state(targets, preds)
    print('accuracy_tf', m.result().numpy())

    m = tf.keras.metrics.Precision()
    m.update_state(targets, preds)
    print('precision_tf', m.result().numpy())

    m = tf.keras.metrics.Recall()
    m.update_state(targets, preds)
    print('recall_tf', m.result().numpy())

    m = tf.keras.metrics.AUC()
    m.update_state(targets, preds)
    print('auc_tf', m.result().numpy())

    if include_confusion_matrix:
        print('confusion matrix (normalised):')
        cm = confusion_matrix(
            targets_manyhot,
            preds_manyhot,
            normalize='true'
        )
        pretty_print_confusion_matrix(cm, labels)
        
        print('confusion matrix:')
        cm2 = confusion_matrix(
            targets_manyhot,
            preds_manyhot
        )
        pretty_print_confusion_matrix(cm2, labels)



def plot_mislabeled(model, dataset, label_dict):
    
    i = 0
    for bx, by in dataset:   # load batches
        
        bpreds = model.predict(bx)
        for p, y, x in zip(bpreds, by, bx):
            
            true_label = label_dict[np.argmax(y)]
            pred_label = label_dict[np.argmax(p)]
            
            if pred_label != true_label:
                i += 1
                plt.imshow(x.numpy().astype('uint8'))
                plt.xticks([])
                plt.yticks([])
                plt.xlabel(f'true: {true_label}, predicted: {pred_label}', fontsize=16)
                plt.subplots_adjust(left=0, right=1, top=1, bottom=0.1)
                plt.show()

                #filename = r'C:\Users\steffen\Dropbox (NORSAR)\ML for building typologies\images\ml-misclassified' + r'\example-' + str(i)
                #plt.savefig(filename, bbox_inches='tight')

                plt.show()






def pretty_print_confusion_matrix(matrix, labels):

    if 'float' in matrix.dtype.name:
        matrix = np.round(matrix, 3)

    
    print('\t\t\t\t\t\tpredicted')
    row = '     {:16}'.format('')
    for l in labels:
        row += '{:>16}'.format(l)
    print(row)

    for irow in range(matrix.shape[0]):
        row = ''
        if irow == 2:
            row += 'true '
        else:
            row += '     '
        row += '{:>16}'.format(labels[irow])

        for icol in range(matrix.shape[1]):
            row += '{:16}'.format(matrix[irow][icol])
        print(row)


def get_files_from_list(filename):

    with open(filename) as fin:
        files = [l.strip() for l in fin.readlines()]

    return files

def create_symlinks(target_dir: Path, files: List[Path]):
    """
    Given two lists of files, create symlinks in the target directory
    """

    if target_dir.exists():
        print('Removing', target_dir)
        rmtree(target_dir)
    
    print('Creating', target_dir)

    Path.mkdir(target_dir, parents=True)
    for t in TYPOLOGIES:
        Path.mkdir(target_dir / t)


    def make_link(f):
        
        f = Path(f)
        f = Path.resolve(f)
        cat = f.parent.__str__().split('/')[-1]
        link = Path(Path.resolve(target_dir) / cat / f.name)
        try:
            link.symlink_to(f)
            return 0
        except FileExistsError:
            return 1

    n_duplicates = 0

    for f in files:
        n_duplicates += make_link(f)

    print(f'Skipped {n_duplicates} duplicate images')
    
    

if __name__ == '__main__':

    parser = ArgumentParser('Train and evaluate model')
    parser.add_argument('-m', '--model_name', type=str, required=True)
    #parser.add_argument('-i', '--input_dir', type=str, required=True)
    parser.add_argument('-t', '--train', action='store_true')
    parser.add_argument('-e', '--evaluate', action='store_true')
    parser.add_argument('-pp', '--plot_predictions', action='store_true')
    args = parser.parse_args()

    #data_dir = Path(args.input_dir)
    #assert data_dir.is_dir(), f'No such directory: {data_dir}'

    # Prepare directory with symlinks to data 
    train_files = np.array(get_files_from_list('imgs_for_training.txt'))
    test_files = np.array(get_files_from_list('imgs_for_testing.txt'))

    train_dir = Path('symlinks/train')
    test_dir = Path('symlinks/test')
    if not train_dir.is_dir():
        create_symlinks(train_dir, train_files)
    if not test_dir.is_dir():
        create_symlinks(test_dir, test_files)

    imgsize = (224, 224)    # ResNet
    #imgsize = (299, 299)    # Inception-ResNet
    imgshape = imgsize + (3,)

    label_names = sorted(TYPOLOGIES)
    label_names = {i: l for i, l in enumerate(label_names)}

    batchsize = 32
    seed = 134
    val_split = 0.2
    
    print('Preparing train data')
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        train_dir,
        image_size=imgsize,
        batch_size=batchsize,
        label_mode='categorical',
        seed=seed,
        validation_split=val_split,
        subset='training',
    )

    print('Preparing validation data')
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        train_dir,
        image_size=imgsize,
        batch_size=batchsize,
        label_mode='categorical',
        seed=seed,
        validation_split=val_split,
        subset='validation',
    )

    print('Preparing test data')
    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        test_dir,
        image_size=imgsize,
        batch_size=batchsize,
        label_mode='categorical'
    )

    model = None

    if args.train:
        model = train(train_ds, val_ds, imgshape, num_classes=len(label_names), save_as=args.model_name)


    if args.plot_predictions:

        if model is None:
            model = tf.keras.models.load_model(args.model_name)

        plot_mislabeled(model, val_ds, label_names)
        """
        # Plot along with predictions
        plt.figure(figsize=(10, 10))
        for images, labels in val_ds.take(1):
            preds = model.predict(images)
            for i in range(images.shape[0]):
                ax = plt.subplot(4, 4, i + 1)
                plt.imshow(images[i].numpy().astype('uint8'))
                true_label = np.argmax(labels[i], axis=0) 
                true_label = label_names[true_label]
                best_pred = np.argmax(preds[i], axis=0)
                best_pred = label_names[best_pred]
                plt.xlabel(f'true: {true_label}, pred: {best_pred}')
                ax.set_yticklabels([])
                ax.set_xticklabels([])
            plt.show()
        """


    if args.evaluate:

        if model is None:
            model = tf.keras.models.load_model(args.model_name)
        
        print('Fine-tuned model metrics:')
        print('')
        print('Train dataset metrics:')
        evaluate(model, train_ds, sorted(label_names.values()))

        print('')
        print('Validation dataset metrics:')
        evaluate(model, val_ds, sorted(label_names.values()))

        print('')
        print('Test dataset metrics:')
        evaluate(model, test_ds, sorted(label_names.values()))

        pre_model = tf.keras.models.load_model(args.model_name + '_pre_finetuning')
        print('')
        print('Pre-fine-tuned model metrics:')
        print('')
        print('Train dataset metrics:')
        evaluate(pre_model, train_ds, sorted(label_names.values()), include_confusion_matrix=False)

        print('')
        print('Validation dataset metrics:')
        evaluate(pre_model, val_ds, sorted(label_names.values()), include_confusion_matrix=False)

        print('')
        print('Test dataset metrics:')
        evaluate(pre_model, test_ds, sorted(label_names.values()), include_confusion_matrix=False)

    