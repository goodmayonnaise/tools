import os, random, shutil

if __name__ == "__main__":
    
    path = r""
    dirs = []

    T = []
    f_path = os.path.join(path, "false")
    F = [os.path.join(f_path, i) for i in os.listdir(f_path)]
    
    for dir in dirs :
        path2 = os.path.join(path, dir)
        dir_nums = os.listdir(path2)
        
        for dir_num in dir_nums:
            path3 = os.path.join(path2, dir_num)
            files = os.listdir(path3)
            js = [i for i in files if i[-4:]=='json']
            if len(js) == 0:
                for f in files:
                    F.append(os.path.join(path3, f))
            else:
                for j in js :
                    # T.append(os.path.join(path3, j)) 
                    T.append(os.path.join(path3, j[:-4]+"tif"))
                    files.remove(j)
                    files.remove(j[:-4]+"tif")
                for f in files:
                    F.append(os.path.join(path3, f))

    # random split
    random.seed(42)
    random.shuffle(T)

    for i in range(len(T)):
        if i < 27 :
            mode = "train"
        elif i < 32 :
            mode = "val"
        else:
            mode = "test"
        shutil.move(T[i], os.path.join(path, "images", mode))
        shutil.move(T[i].replace("tif", "json"), os.path.join(path, "annotations", mode))

    random.shuffle(F)
    for i in range(len(F)):
        if i < 90:
            mode = "train"
        elif i < 110:
            mode = "val"
        else:
            mode = "test"
        shutil.move(F[i], os.path.join(path, "images", mode, F[i].split("\\")[-1]))



                    
                    

                    


            
            
