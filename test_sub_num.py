import glob
import os
import openslide

train_dir = '/media/huoy1/48EAE4F7EAE4E264/Projects/detection/data/kidpath_multiROI/train'
val_dir = '/media/huoy1/48EAE4F7EAE4E264/Projects/detection/data/kidpath_multiROI/val'
test_dir = '/media/huoy1/48EAE4F7EAE4E264/Projects/detection/data/kidpath_multiROI/test'


images = glob.glob(os.path.join(test_dir, '*.png'))
strlist = []
for fi in range(len(images)):
    os.path.basename(images[fi])
    strlist.append('%s%s' %(os.path.basename(images[fi]).split('-x-')[0],os.path.basename(images[fi]).split('-x-')[1]))

print(len(set(strlist)))

source_dir = '/media/huoy1/48EAE4F7EAE4E264/Projects/from_haichun/batch_1_data/scn'

sublist = {}
sublist['train'] = ['Case 01', 'Case 02', 'Case 06', 'Case 08', 'Case 14', 'Case 15', 'Case 17', 'Case 19', 'Case 22', 'Case 23', 'Case 24', 'Case 25']
sublist['validation'] = ['Case 11', 'Case 12', 'Case 18', 'Case 20']
sublist['test'] = ['Case 03', 'Case 05', 'Case 09', 'Case 16']

dtype = 'test'

good_count = 0
for si in range(len(sublist[dtype])):
    subname = sublist[dtype][si]
    files = glob.glob(os.path.join(source_dir, '%s*.scn' % subname))
    for fi in range(len(files)):
        scn_file = files[fi]
        try:
            simg = openslide.open_slide(scn_file)
            good_count = good_count+1
            print(' read %s, si = %d, good count = %d' % (scn_file, si+1, good_count))
        except:
            print('can not read %s' % scn_file)
            continue



