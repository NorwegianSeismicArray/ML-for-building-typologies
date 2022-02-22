import os
from urllib import request
import json
from typing import Tuple
import csv
from argparse import ArgumentParser
from pyproj import Geod


API_KEY = os.environ['GMAPS_API_KEY']
assert API_KEY != ''


def download_image(coordinates: Tuple[float, float], filename: str, geod: Geod):
    """
    For a given set of coordinates, construct a StreetView URL and download
    it as 'filename'

    Coordinates are in WGS84 (lat/lon)
    """

    assert len(coordinates) == 2, f'Bad coordinates: {coordinates}'

    meta_url = 'https://maps.googleapis.com/maps/api/streetview/metadata?'
    image_url = 'https://maps.googleapis.com/maps/api/streetview?'

    # Download metadata to check if image is available
    max_radius = '40'   # Image must be taken within this distance from the given coordinates
    params = 'location={},{}&radius={}&source=outdoor&key={}'.format(
        coordinates[0],
        coordinates[1],
        max_radius,
        API_KEY
    )

    meta = request.urlopen(meta_url + params).read()
    meta = json.loads(meta)
    if meta['status'] != 'OK':
        return meta['status']

    # Compute distance between building and photo location
    #_, _, dist = geod.inv(coordinates[0], coordinates[1], float(meta['location']['lat']), float(meta['location']['lng']))

    # StreetView image parameters: (all must be strings)
    imgsize = '456x456'
    # Field of view -- above 90 means "fisheye"
    field_of_view = 100 # str(round(3000/dist))
    # Upwards angle
    pitch = '15'

    params = 'location={},{}&size={}&fov={}&pitch={}&radius={}&source=outdoor&key={}'.format(
        coordinates[0],
        coordinates[1],
        imgsize,
        field_of_view,
        pitch,
        max_radius,
        API_KEY
    )

    # Download image
    request.urlretrieve(image_url + params, filename)

    return meta['status']



def download_from_csv(input_csv, output_dir, x_coord_name, y_coord_name, id_column_name):

    if not os.path.isdir(output_dir):
        print(f'Creating {output_dir}')
        os.mkdir(output_dir)

    geodesic = Geod(ellps='WGS84')

    
    #input_csv_name = 'building_numbers_and_coords_in_area1.csv'
    output_csv_name = 'downloaded_images.csv'
    #output_path = '../images/StreetView/in_area_1'

    # Extract header from input CSV file
    input_csv = open(input_csv, 'r')
    header = input_csv.readline().strip().split(',')
    input_csv.seek(0)

    # Output CSV will have same header, but including filename and status
    header.append('status')
    header.append('filepath')

    output_csv = open(output_csv_name, 'w', newline='')
    csv_writer = csv.DictWriter(output_csv, fieldnames=header)
    csv_writer.writeheader()

    print('Downloading...')

    # Read input and process
    csv_reader = csv.DictReader(input_csv, delimiter=',')
    for row in csv_reader:

        coords = (float(row[x_coord_name]), float(row[y_coord_name]))

        # For building lists
        if id_column_name == 'building_number':
            bldg_num = row['building_number']
            outpath = os.path.join(output_dir, f'bldg_number_{bldg_num}.jpg')
        # For other uses
        else:
            id_num = row[id_column_name]
            outpath = os.path.join(output_dir, f'id_number_{id_num}.jpg')
            

        status = download_image(coords, outpath, geodesic)

        out_row = dict(row)
        out_row['status'] = status
        out_row['filepath'] = '-'

        if status == 'OK':
            out_row['filepath'] = outpath
        
        csv_writer.writerow(out_row)

        print(f'\r{outpath} - {status}', end='')
    
    print()
    print('Done.')

 
    # Test a single image
    #download_image((59.96163183, 10.74712747), 'testimage.jpg', geodesic)
    #download_image((59.91872655435505, 10.74923624977102), 'testimage.jpg', geodesic)

    

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('input_csv', type=str, help='Input CSV')
    parser.add_argument('output_path', type=str, help='Output directory')
    parser.add_argument('-cx', '--coord_x', type=str, default='coords_WGS84:x', help='Name of X coord column in csv file')
    parser.add_argument('-cy', '--coord_y', type=str, default='coords_WGS84:y', help='Name of Y coord column in csv file')
    parser.add_argument('-id', '--id_col', type=str, default='building_number', help='Name of ID column')

    args = parser.parse_args()

    download_from_csv(args.input_csv, args.output_path, args.coord_x, args.coord_y, args.id_col)


