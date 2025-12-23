import os
import random 
import shutil

if __name__ == "__main__":
    # path = r''
    path = r''
    datas = os.listdir(path)
    
    img, js = [], []
    for d in datas:
        if 'tif' in d:
            img.append(d)
        else:
            js.append(d)
            
    random.seed(42)
    src = r""
    name = [i[:-3] for i in img]
    random.shuffle(name)
    for i in range(len(name)):
        # if i < 90:
        #     shutil.move(os.path.join(path, name[i]) + "json", os.path.join(src, "train", "false", name[i])+"json")
        #     shutil.move(os.path.join(path, name[i]) + "tif", os.path.join(src, "train", "false", name[i])+"tif")
        # elif i < 110:
        #     shutil.move(os.path.join(path, name[i]) + "json", os.path.join(src, "val", "false", name[i])+"json")
        #     shutil.move(os.path.join(path, name[i]) + "tif", os.path.join(src, "val", "false", name[i])+"tif")
        # else:
        #     shutil.move(os.path.join(path, name[i]) + "json", os.path.join(src, "test", "false", name[i])+"json")
        #     shutil.move(os.path.join(path, name[i]) + "tif", os.path.join(src, "test", "false", name[i])+"tif")

        if i < 27:
            shutil.move(os.path.join(path, name[i]) + "json", os.path.join(src, "train", "true", name[i])+"json")
            shutil.move(os.path.join(path, name[i]) + "tif", os.path.join(src, "train", "true", name[i])+"tif")
        elif i < 32:
            shutil.move(os.path.join(path, name[i]) + "json", os.path.join(src, "val", "true", name[i])+"json")
            shutil.move(os.path.join(path, name[i]) + "tif", os.path.join(src, "val", "true", name[i])+"tif")
        else:
            shutil.move(os.path.join(path, name[i]) + "json", os.path.join(src, "test", "true", name[i])+"json")
            shutil.move(os.path.join(path, name[i]) + "tif", os.path.join(src, "test", "true", name[i])+"tif")
        
