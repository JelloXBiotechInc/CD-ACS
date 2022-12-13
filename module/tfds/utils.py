import os, sys, json
from .get_image_size import get_image_size

class Utils:
    @staticmethod
    def get_gdrive_download_url(file_hash):
        return f'https://drive.google.com/uc?export=download&id={file_hash}&confirm=t'
    
    @staticmethod
    def extract_files_base_on_folder_from_extracted_dir(extracted_dir):
        data = {}
        print(f'***extracted_dir : {extracted_dir}\n\n')
        for img_path in extracted_dir.glob('*/*'):
            splitted = str(img_path).split(os.sep)
            data_type, name = splitted[-2:]
            if data_type not in data:
                data[data_type] = []
            data[data_type].append({
                'file_name': name.split('/')[-1],
                'f_obj': img_path,
            })
        return data
    
    @staticmethod
    def compare_img_and_target(img, target):
        def bname(t):
            return '.'.join(t.split('.')[:-1])
        
        def rstrip(t, sub):
            if t[len(t) - len(sub):] == sub:
                return t[:len(t) - len(sub)]
            return t
        
        def process(t):
            t = rstrip(t, '.jellox')
            t = rstrip(t, '_t')
            t = rstrip(t, '-mask')
            return t
        
        basename_img = bname(img['file_name'])
        if basename_img == 'Thumbs':
            return False
        basename_target = bname(target['file_name'])
        if process(basename_img) == process(basename_target):
            return True
        return False
    
    @staticmethod
    def pair_files_from_2_list(a_list, b_list, cmp_callback=None):
        cmp_callback = cmp_callback if cmp_callback else Utils.compare_img_and_target
        
        data_pairs = []
        pairs = []
        for img in a_list:
            found = None
            for target in b_list:
                if cmp_callback(img, target):
                    found = target
            if found:
                target = found
                pairs.append({
                    'file_name': img['file_name'],
                    'segmentation_file_name': target['file_name'],
                })
                data_pairs.append({
                    'file_name': img['file_name'],
                    'image': img['f_obj'],
                    'segmentation_mask': target['f_obj'],
                })
        print(json.dumps(pairs, indent=4), len(data_pairs))
        
        return data_pairs
    
    @staticmethod
    def pair_files_from_3_list(a_list, b_list, c_list, cmp_callback=None):
        cmp_callback = cmp_callback if cmp_callback else Utils.compare_img_and_target
        
        data_pairs = []
        pairs = []
        for img in a_list:
            found1 = None
            found2 = None
            for target in b_list:
                if cmp_callback(img, target):
                    found1 = target
            for target in c_list:
                if cmp_callback(img, target):
                    found2 = target
            
            if found1 and found2:
                target1 = found1
                target2 = found2
                pairs.append({
                    'file_name': img['file_name'],
                    'segmentation_file_name': target1['file_name'],
                    'color_region_file_name': target2['file_name'],
                })
                data_pairs.append({
                    'file_name': img['file_name'],
                    'image': img['f_obj'],
                    'segmentation_mask': target1['f_obj'],
                    'color_region_mask': target2['f_obj'],
                })
        print(json.dumps(pairs, indent=4), len(data_pairs))
        
        return data_pairs
    
    @staticmethod
    def get_image_size(f):
        return get_image_size(str(f))