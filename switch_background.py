from PIL import Image
import os
import numpy as np
path = r'C:\Users\asus\Project\Nutriobot\Dataset\fruits-360_dataset\fruits-360\Training'
dest = r'C:\Users\asus\Project\Nutriobot\Dataset\fruits-360-random-bg-v2\Training'


def switch_bg(root_dir, dest):
    root_dir_result = dest
    fruit_dirs = os.listdir(root_dir)
    fruit_num = 0
    total_fruit = len(fruit_dirs)
    # iterating through each fruit directory
    for fruit_dir in fruit_dirs:
        fruit_name = fruit_dir
        fruit_num += 1
        abs_fruit_dir = os.path.join(root_dir_result, fruit_dir)
        if not os.path.isdir(abs_fruit_dir):
            os.mkdir(abs_fruit_dir)
        fphoto_list = os.listdir(os.path.join(root_dir, fruit_dir))
        total_img = len(fphoto_list)
        i = 1
        # iteratic through each images in a fruit directory
        for fphoto in fphoto_list:
            fruit_img = Image.open(os.path.join(root_dir,
                                                fruit_dir, fphoto)).convert('RGBA')
            datas = fruit_img.getdata()
            new_img_data = []
            os.system('cls')
            print('Processing fruit {} of {} ({})'.format(
                fruit_num, total_fruit, fruit_name))
            print('Processing image {} of {} '.format(i, total_img))
            # detecting white pixels and replaing with tranparent color
            for item in datas:
                if item[0] >= 210 and item[1] >= 210 and item[2] >= 210:
                    new_img_data.append((item[0], item[1], item[2], 0))
                else:
                    new_img_data.append(item)
            fore_img = Image.new("RGBA", (fruit_img.width, fruit_img.height))
            fore_img.putdata(new_img_data)
            # generate random bg
            red = np.random.randint(0, 255)
            green = np.random.randint(0, 255)
            blue = np.random.randint(0, 255)
            random_color = (red, green, blue)
            bg_img = Image.new(mode='RGBA', size=(
                100, 100), color=random_color)
            bg_img.paste(fore_img, (0, 0), fore_img)
            bg_img.save(format_name(abs_fruit_dir, fruit_name, i))
            i += 1
        print('Done processing fruit {} of {}'.format(fruit_num, total_fruit))


def format_name(path, fruit_name, i):
    fruit_name = fruit_name.replace(' ', '_')
    return os.path.join(path, (fruit_name+'-'+str(i)+'.png'))


switch_bg(path, dest)
