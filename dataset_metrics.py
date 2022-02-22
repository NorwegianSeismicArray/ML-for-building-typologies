from pathlib import Path


if __name__ == '__main__':

    # Load all CSV files 
    input_csv_files = Path('csv').glob('*')
    image_top_dir = Path('images')

    # Load all image file names
    img_numbers = []
    subdirs = image_top_dir.glob('*')
    for sdir in subdirs:

        imgs = sdir.glob('bldg*.jpg')
        for imgname in imgs:
            num = imgname.name.replace('bldg_number_', '').replace('.jpg', '')
            img_numbers.append(int(num))

    img_numbers = set(img_numbers)
    print(f'Read {len(img_numbers)} image numbers')


    # Load building numbers
    all_bldg_numbers = []
    for csv_file in input_csv_files:
        
        bldg_numbers = []
        with open(csv_file) as fin:
            header = fin.readline()
            for line in fin.readlines():
                num = int(line.split(',')[0])
                bldg_numbers.append(num)
        
        all_bldg_numbers += bldg_numbers
        
        # Print metrics for this area
        bldg_numbers = set(bldg_numbers)
        missing_imgs = bldg_numbers - img_numbers

        print('')
        print(csv_file)
        missing_ratio = len(missing_imgs)/len(bldg_numbers)*100.0
        print('Missing images: {:.3f} % (coverage {:.3f} %) ({} / {})'.format(
            missing_ratio,
            100.0 - missing_ratio,
            len(missing_imgs),
            len(bldg_numbers)
        ))

    print('')
    print(f'Read {len(all_bldg_numbers)} building numbers')
    
    all_bldg_numbers = set(all_bldg_numbers)

    missing_imgs = all_bldg_numbers - img_numbers
    missing_ratio = len(missing_imgs)/len(all_bldg_numbers)*100.0

    print('Missing images: {:.3f}% (coverage {:.3f} %) ({} / {})'.format(
        missing_ratio,
        100.0 - missing_ratio,
        len(missing_imgs),
        len(all_bldg_numbers)
    ))
    surplus_imgs = img_numbers - all_bldg_numbers
    print('Surplus images: {}'.format(len(surplus_imgs)))

