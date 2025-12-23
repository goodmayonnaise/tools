import os, shutil

if __name__ == "__main__":
    path = r""
    dirs = []
    for dir in dirs:
        dirnums = os.listdir(os.path.join(path, dir))
        for dirnum in dirnums:
            fnames = os.listdir(os.path.join(path, dir, dirnum))
            
            for fname in fnames:
                shutil.move(os.path.join(path, dir, dirnum, fname), os.path.join(path, dir, dirnum, f"{dir}_{fname}"))
