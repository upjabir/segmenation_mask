import numpy as np
import json
import cv2
from pathlib import Path
from PIL import Image
import argparse



def main(input_path , out_path):
    main_data = Path(input_path)
    mask_data = Path(out_path)
    try:
        mask_data.mkdir(parents=True , exist_ok=False)
        print('Mask Folder created')
    except:
        print("Mask Folder Already exists")

    for p in main_data.glob('**/*.json'):
        posix_path = str(p).split('.')[0]
        posix_path = posix_path.split('/')[1:-1]
        posix_path = '/'.join(posix_path)
        print(posix_path)
        
        with open(str(p) , 'r') as read_file:
            data = json.load(read_file)

        all_file_names = list(data.keys())

        for j in range(len(all_file_names)):
            image_name=data[all_file_names[j]]['filename']
            print("Image Name :",str(main_data / posix_path /image_name))
            img = np.asarray(Image.open(str(main_data / posix_path /image_name)))

            if data[all_file_names[j]]['regions'] != {}:
                print("File Number inside json:",j)

                try:
                    shape1_x=data[all_file_names[j]]['regions']['0']['shape_attributes']['all_points_x']
                    shape1_y=data[all_file_names[j]]['regions']['0']['shape_attributes']['all_points_y']
                except : 
                    shape1_x=data[all_file_names[j]]['regions'][0]['shape_attributes']['all_points_x']
                    shape1_y=data[all_file_names[j]]['regions'][0]['shape_attributes']['all_points_y']


                ab=np.stack((shape1_x, shape1_y), axis=1)
                mask = np.zeros((img.shape[0],img.shape[1]))
                img2=cv2.drawContours(mask.copy(), [ab], -1, 255, -1)
                mask_save_path = mask_data /posix_path
                try:
                    mask_save_path.mkdir(parents=True,exist_ok=False)
                    print("Mask sub folder created")
                except:
                    print("Mask sub Folder already exist")

                cv2.imwrite(str(mask_save_path /image_name),img2.astype(np.uint8))
            else:
                print("Region not found")
        

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--data_path", required=True, help="Path To Input Images And Annotation")
    ap.add_argument("-o", "--output_path", required=True, help="Path To Output Path")
    args = vars(ap.parse_args())
    
    main(args['data_path'] , args['output_path'])
