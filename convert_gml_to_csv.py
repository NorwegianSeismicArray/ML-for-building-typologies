"""
Take GML files from geonorge and convert coordinates, plus intersect with
shapefile defining Area 1

(This is superseeded now)
"""



from xml.etree import ElementTree
from csv import DictWriter
from pyproj import Transformer
import shapefile
from shapely.geometry import Polygon, Point



def convert(infile, outfile):

    # Coordinate transformer
    transformer = Transformer.from_crs('epsg:25832', 'epsg:4326')

    itr = ElementTree.iterparse(infile)

    blg_dict = {
        'building_number': None,
        'building_status': None,
        'coords_EPSG_25832:x': None,
        'coords_EPSG_25832:y': None,
        'coords_WGS84:x': None,
        'coords_WGS84:y': None,
    }

    # Output file
    outfile = open(outfile, 'w', newline='')
    csv_writer = DictWriter(outfile, fieldnames=list(blg_dict))
    csv_writer.writeheader()

    # Intersect with first polygon defined in shape file
    with shapefile.Reader('../GIS_data/buildings/test_area.shp') as shape_reader:
        shape = shape_reader.shape(0)
        polygon = Polygon(shape.points)


    for event, elem in itr:

        blg_dict.clear()

        if event == 'end' and elem.tag.endswith('Bygning'):
            
            for sub_elem in elem:

                if sub_elem.tag.endswith('bygningsnummer'):
                    blg_dict['building_number'] = int(sub_elem.text)
                
                if sub_elem.tag.endswith('bygningsstatus'):
                    blg_dict['building_status'] = sub_elem.text

                if sub_elem.tag.endswith('representasjonspunkt'):
                    
                    for pos_elem in sub_elem.iter():
                        if pos_elem.tag.endswith('pos'):
                            coords = pos_elem.text.split()
                            blg_dict['coords_EPSG_25832:x'] = float(coords[0])
                            blg_dict['coords_EPSG_25832:y'] = float(coords[1])

                            blg_dict['coords_WGS84:x'], blg_dict['coords_WGS84:y'] = transformer.transform(
                                blg_dict['coords_EPSG_25832:x'],
                                blg_dict['coords_EPSG_25832:y']
                            )

            point = Point(blg_dict['coords_WGS84:y'], blg_dict['coords_WGS84:x'])
            if polygon.contains(point):
                csv_writer.writerow(blg_dict)

            elem.clear()

    outfile.close()


if __name__ == '__main__':

    #convert('testfile.gml', 'out.out')

    input_gml_file = '../GIS_data/buildings/Basisdata_0301_Oslo_25832_MatrikkelenBygning_GML/Basisdata_0301_Oslo_25832_MatrikkelenBygning_GML.gml'
    output_csv_file = 'building_numbers_and_coords_in_area1.csv'

    convert(input_gml_file, output_csv_file)
    print('Done.')
