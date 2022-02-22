"""
Process the new building list (which includes floor numbers),
convert coordinates and intersect with shapefiles
"""

import os
from csv import DictWriter, DictReader
from pyproj import Transformer
import shapefile
from shapely.geometry import Polygon, Point


bydel_dict = {
    'Gamle_Oslo': '030101',
    'Gruenerloekka': '030102',
    'Sagene': '030103',
    'StHanshaugen': '030104',
    'Frogner': '030105',
    'Ullern': '030106',
    'Vestre_Aker': '030107',
    'Nordre_Aker': '030108',
    'Bjerke': '030109',
    'Grorud': '030110',
    'Stovner': '030111',
    'Alna': '030112',
    'Oestensjoe': '030113',
    'Nordstrand': '030114',
    'Soendre_Nordstrand': '030115',
    'Sentrum': '030116',
    'Nordmarka_Oestmarka': '030117'
}

def get_bydel_shape(bydel):

    bydel_shapefile = 'C:/Users/steffen/Dropbox (NORSAR)/ML for building typologies/GIS_data/bydel-shapefiles/Bydeler.shp'
    bydel_projectionfile = 'C:/Users/steffen/Dropbox (NORSAR)/ML for building typologies/GIS_data/bydel-shapefiles/projection_info.txt'

    projection = ''
    with open(bydel_projectionfile) as prjf:
        projection = prjf.readline().strip().split()[-1]
    assert projection.startswith('EPSG')

    bydelsnr = bydel_dict[bydel]

    with shapefile.Reader(bydel_shapefile) as shapereader:
        for record_num, record in enumerate(shapereader.records()):
            if record.bydelsnr == bydelsnr:
                return shapereader.shape(record_num), projection.lower()


def process_csv(infile, outfile, bydel):

    # Polygon representing bydel borders
    bydel_shape, bydel_proj = get_bydel_shape(bydel)
    bydel_polygon = Polygon(bydel_shape.points)

    # Convert to WGS84 
    transform_matrikkel = Transformer.from_crs('epsg:25832', 'epsg:4326')
    transform_bydel = Transformer.from_crs('epsg:25832', bydel_proj)


    output_fields = {
        'building_number': None,
        'building_status': None,
        'building_type': None,
        'num_floors': None,
        'coords_EPSG_25832:x': None,
        'coords_EPSG_25832:y': None,
        'coords_WGS84:x': None,
        'coords_WGS84:y': None,
    }

    # Output file
    output_file = open(outfile, 'w', newline='')
    csv_writer = DictWriter(output_file, fieldnames=list(output_fields))
    csv_writer.writeheader()

    with open(infile, newline='') as csv_in:
        csv_reader = DictReader(csv_in)
        for row in csv_reader:

            outrow = dict(output_fields)
            outrow['building_number'] = row['bygningsnummer']
            outrow['building_status'] = row['bygningsstatus']
            outrow['building_type'] = row['bygningstype']
            outrow['num_floors'] = row['Etaje']
            outrow['coords_EPSG_25832:x'] = row['Longitude']
            outrow['coords_EPSG_25832:y'] = row['Latitude']

            outrow['coords_WGS84:x'], outrow['coords_WGS84:y'] = transform_matrikkel.transform(
                row['Longitude'],
                row['Latitude']
            )

            # Transform to coordinates expected by bydel shapefile
            point_bydel = Point(
                transform_bydel.transform(
                    row['Longitude'],
                    row['Latitude']
                )
            )
            if bydel_polygon.contains(point_bydel):
                csv_writer.writerow(outrow)
    
    output_file.close()
    print('Wrote file', outfile)



if __name__ == '__main__':

    for bydel in bydel_dict:

        csv_path = os.path.join(
            os.environ.get('userprofile'),
            'Dropbox (NORSAR)/ML for building typologies/GIS_data/buildings/MatrikkelVyg_etaje-coordinates_Oslo.csv'
        )
        assert os.path.exists(csv_path), f'File not found: {csv_path}'

        process_csv(
            csv_path,
            f'csv/building-list-{bydel}.csv',
            bydel
        )

