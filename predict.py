from pathlib import Path
import csv
from argparse import ArgumentParser
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt



def predict():
    pass


def format_prediction(pred, label_dict):

    out = []
    for i in range(len(label_dict)):
        out.append('{}: {:.2f}'.format(label_dict[i], pred[i]))
    
    return ' '.join(out)


if __name__ == '__main__':

    parser = ArgumentParser('Predict typologies')
    parser.add_argument('csv_in', type=str, help='Input CSV file')
    parser.add_argument('csv_out', type=str, help='Output CSV file')
    parser.add_argument('-p', '--plot', action='store_true', help='Plot images')
    args = parser.parse_args()

    model_file = 'densenet201-model-jan11-conv5_take3'
    
    image_path = Path('images')

    input_csv_name = Path(args.csv_in)
    output_csv_name = Path(args.csv_out)
    district = input_csv_name.name.replace('.csv', '').split('-')[-1]

    image_path = image_path/district
    assert image_path.is_dir(), f'No such directory: {image_path}'

    print('Input CSV file:\t\t', input_csv_name)
    print('Output CSV file:\t', output_csv_name)
    print('Model file:\t\t', model_file)
    print('District:\t\t', district)
    print('Image directory:\t', image_path)

    image_size = (224, 224)
    #image_size = (299, 299)
    model = tf.keras.models.load_model(model_file)
    label_names = sorted(['CR', 'MUR', 'other', 'S', 'SRC', 'T'])
    label_names = {i: l for i, l in enumerate(label_names)}

    csv_in = open(input_csv_name, newline='')
    csv_out = open(output_csv_name, 'w', newline='')
    reader = csv.reader(csv_in, delimiter=',')
    writer = csv.writer(csv_out, delimiter=',')
    
    header = next(reader)
    header.append('predicted_type')
    for i in range(len(label_names)):
        header.append('prob_{}'.format(label_names[i]))
    writer.writerow(header)

    i = 0
    for row in reader:

        bldg_num = int(row[0])
        image_name = image_path / f'bldg_number_{bldg_num}.jpg'
        if not image_name.exists():
            output = ['-'] * (len(label_names) + 1)
        else:
            img = tf.keras.preprocessing.image.load_img(image_name, target_size=image_size)
            x = tf.keras.preprocessing.image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            pred_num = model.predict(x)[0]
            pred_max = np.argmax(pred_num, axis=0)
            pred = label_names[pred_max]

            if args.plot:
                plt.imshow(img)
                plt.xlabel(format_prediction(pred_num, label_names))
                plt.show()

            output = [pred] + list(pred_num)
        
        out_row = list(row)
        out_row += output
        writer.writerow(out_row)


    csv_in.close()
    csv_out.close()



