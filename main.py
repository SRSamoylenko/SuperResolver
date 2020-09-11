import cv2
from screen import Screen
from instrumental import instrument
import json
from datetime import datetime
import screeninfo
import os
import numpy as np
from tqdm import tqdm

if __name__ == '__main__':
    print('Captures per mask: ')
    captures_per_mask = int(input())
    date = datetime.today().strftime('%Y-%m-%d %H.%M.%S')
    slm_description = screeninfo.get_monitors()[1]
    print('Second monitor has been found.\n')

    with open('camera_params.json', 'r') as f:
        camera_params = json.load(f)
    print('Camera parameters have been imported.\n')

    with open('camera_serial.json', 'r') as f:
        camera_serial = bytes(json.load(f), encoding='utf-8')
    camera = instrument(serial=camera_serial)
    print('Camera has been opened.\n')

    with open('mask_list.json', 'r') as f:
        mask_list = json.load(f)
    print('Masks have been imported.\n')

    with open('screen_params.json', 'r') as f:
        screen_params = json.load(f)
    print('Screen parameters have been imported.\n')

    with open('source_list.json', 'r') as f:
        source_list = json.load(f)
    print('List of sources has been imported.\n')

    os.mkdir(date)
    os.chdir(date)

    with open('additional_parameters.json', 'w') as f:
        json.dump({
            'slm_resolution': (slm_description.width, slm_description.height),
            'captures_per_mask': captures_per_mask
        }, f, indent=4)

    with open('camera_params.json', 'w') as f:
        json.dump(camera_params, f, indent=4)

    with open('mask_list.json', 'w') as f:
        json.dump(mask_list, f, indent=4)

    with open('screen_params.json', 'w') as f:
        json.dump(screen_params, f, indent=4)

    with open('source_list.json', 'w') as f:
        json.dump(source_list, f, indent=4)

    screen = Screen(resolution=(slm_description.width, slm_description.height), source_list=source_list,
                    mask_list=mask_list, **screen_params)
    print('Screen has been initialized.\n')

    cv2.namedWindow('image', cv2.WND_PROP_FULLSCREEN)
    cv2.moveWindow('image', x=slm_description.x, y=slm_description.y)
    cv2.setWindowProperty('image', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    print('Window has been initialized.\n')

    image_counter = 0
    while screen.update_mask():
        print("Mask {} of {}.\n".format(screen.current_mask_number, len(mask_list)))
        for i in tqdm(range(captures_per_mask)):
            image_counter += 1
            cv2.imshow('image', screen.update_screen().astype(np.uint8))
            key = cv2.waitKey(1)
            if key == 27:
                break
            image = camera.grab_image(**camera_params)
            image_name = str(image_counter) + '_' \
                         + screen.mask_list[screen.current_mask_number - 1]['mode_params']['mode_type'] \
                         + '_' + str(screen.mask_list[screen.current_mask_number - 1]['mode_params']['order']) \
                         + '_' + str(i+1) + '.bmp'
            cv2.imwrite(image_name, image)
        print('\n')
    print('Measurement has been successfully finished.\n')

    #
    # camera = instrument(camera_serial)
    # screen = Screen(**screen_params)
    #
    # for source in sources_list:
    #     screen.source_plane.add_source(**source)
    #
    # while screen.update_mask():
    #     for i in range(captures_per_mask):
    #         cv2.
    #             screen.update_screen()
    #         tmp_img = camera.grab_image(**camera_params)
    #
    # camera.close()
    # os.mkdir(date)
    # os.chdir(date)
