import os
from pathlib import Path
from argparse import ArgumentParser
from PIL import Image, ImageOps



def crop_image(img):

    #img.show()

    width, height = img.size

    if height > width:
        excess = (height - width) // 2
        new_size = (0, 0+excess, width, height-excess)
    elif height < width:
        excess = (width - height) // 2
        new_size = (0+excess, 0, width-excess, height)
    else:
        return img
    
    cropped = img.crop(new_size)
    #cropped.show()

    #input(f'{cropped.size}')

    return cropped


def downsample_image(img):
    
    size = (456, 456)
    return img.resize(size)


def verify_images(path):

    for f in iglob(path + '/*/*.jpg'):
        img = Image.open(f)
        img.verify()
        print(f'{f} ok')



if __name__ == '__main__':


    #default_source_dir = Path(os.environ.get('userprofile')) / Path(r'Dropbox (NORSAR)\ML for building typologies\images\images_for_ML_final')
    default_source_dir = None
    
    #destination_dir = Path(r'C:\Users\steffen\projects\ml-for-building-typologies\copied_images\june21')
    typologies = ['CR', 'MUR', 'S', 'SRC', 'T', 'other']

    parser = ArgumentParser('Process and copy images')
    parser.add_argument('destination', type=str, help='Output destination directory')
    parser.add_argument('--source', type=str, default=str(default_source_dir), help='Source directory')
    args = parser.parse_args()

    destination_dir = Path(args.destination)
    source_dir = Path(args.source)
    assert source_dir.is_dir(), f'No such directory: {source_dir}'

    print(f'Copying from {source_dir}')
    print(f'          to {destination_dir}')

    # Create output directory structure
    if not destination_dir.exists():
        Path.mkdir(destination_dir)
    for typ in typologies:
        typdir = destination_dir / typ
        if not typdir.exists():
            Path.mkdir(typdir)

    # Loop over all images 
    for source in source_dir.glob('*'):
        #print('copying from', source.name)
        for typ_dir in source.glob('*'):
            print('  ', typ_dir)
            if not typ_dir.is_dir():
                continue

            typ = typ_dir.name
            if typ not in typologies:
                print('Skipping:', typ_dir)
                continue
            
            outdir = destination_dir / typ

            for filename in typ_dir.glob('*'):


                outname = outdir / filename.name

                image = Image.open(filename)

                # Get the correct orientation (if camera was rotated)
                image = ImageOps.exif_transpose(image)

                size = image.size 
                if size[0] != size[1]:
                    image = crop_image(image)
                    image = downsample_image(image)

                image.save(outname)
                #input(f'saved image as {outname}')



