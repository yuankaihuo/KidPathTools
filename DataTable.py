import openslide
import pandas as pd
import xmltodict
import os
import glob

source_dir = '/Volumes/Yuzhe_Disk/Pathology/R24 scan slides'
output_dir = '/Volumes/Yuzhe_Disk/Pathology/Output_new'

#output csv classification
output_csv_dir = '/Volumes/Yuzhe_Disk/Pathology/csv_dir'

if not os.path.exists(output_csv_dir):
    os.makedirs(output_csv_dir)

output_csv_file = os.path.join(output_csv_dir, 'output.csv')

xml_files = glob.glob(os.path.join(source_dir,'*.xml'))
xml_files.sort()

scn_files = glob.glob(os.path.join(source_dir,'*.scn'))
scn_files.sort()

df = pd.DataFrame(columns=['image_name', 'path', 'label', 'label_name'])
row = 0

for i in range(len(xml_files)):
    xml_file = xml_files[i]
    basename = os.path.basename(xml_file)
    fname, surfix = os.path.splitext(basename)
    xml_file = os.path.join(source_dir, fname + '.xml')

    with open(xml_file) as fd:
        doc = xmltodict.parse(fd.read())
        layers = doc['Annotations']['Annotation']

    try:
        multi_contours = layers['Regions']

        if(len(multi_contours)<2):
            notFound = multi_contours[0]

        else:
            try:
                label_name = layers['Attributes']['Attribute']['@Name']
            except:
                label_name = 'unknown'

            contours = multi_contours['Region']

            try:
                contours['Vertices']
                file_name = fname + '-x-' + label_name + '-x-ROI-1'
                image_name = file_name + '.jpg'
                file_path = os.path.join(output_dir, fname, image_name)
                label = 0
                df.loc[row] = [file_name, file_path, label, label_name]
                row = row + 1

            except:
                for k in range(len(contours)):
                    file_name = fname + '-x-' + label_name + '-x-ROI-' + str(k + 1)
                    image_name = file_name + '.jpg'
                    file_path = os.path.join(output_dir, fname, image_name)
                    label = 0
                    df.loc[row] = [file_name, file_path, label, label_name]
                    row = row + 1


    except:
        for i in range(len(layers)):

            contours = layers[i]['Regions']

            if(len(contours)<2):
                notFound = layers[0]

            else:
                try:
                    label_name = layers[i]['Attributes']['Attribute']['@Name']
                except:
                    label_name = 'unknown'

                contours = contours['Region']

                try:
                    contours['Vertices']
                    file_name = fname + '-x-' + label_name + '-x-ROI-1'
                    image_name = file_name + '.jpg'
                    file_path = os.path.join(output_dir, fname, image_name)
                    label = 0
                    df.loc[row] = [file_name, file_path, label, label_name]
                    row = row + 1

                except:
                    for j in range(len(contours)):
                        file_name = fname + '-x-' + label_name + '-x-ROI-' + str(j + 1)
                        image_name = file_name + '.jpg'
                        file_path = os.path.join(output_dir, fname, image_name)
                        label = 0
                        df.loc[row] = [file_name, file_path, label, label_name]
                        row = row + 1

df.to_csv(output_csv_file, index=False)
