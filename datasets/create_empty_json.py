import os, json, cv2
from glob import glob
from collections import OrderedDict

if __name__ == "__main__":
    path = ""
    dirs = os.listdir(path)
    for dir in dirs:
        path2 = os.path.join(path, dir)
        dirs2 = os.listdir(path2)
        for dir2 in dirs2:
            path3 = os.path.join(path2, dir2)
            fnames =  os.listdir(os.path.join(path2, dir2))
            # print(fnames)
            test = glob(os.path.join(path3, "*.json"))
            if len(test) > 0:
                jnames ={os.path.splitext(fname)[0] for fname in fnames if fname.lower().endswith(".json")}
                fnames = [fname for fname in fnames if not (os.path.splitext(fname)[0] in jnames and fname.lower().endswith((".json", ".tif")))]
            # print(fnames)
            # print()
            for fname in fnames:
                path4 = os.path.join(path3, fname)
                img = cv2.imread(path4)
                h, w, _ = img.shape 
                file_data = OrderedDict()
                file_data["version"] = "5.10.1"
                file_data["flags"] = {}
                file_data["shape"] = []
                file_data["imagePath"] = fname
                file_data["iamgeData"] = ""
                file_data["imageHigth"] = h
                file_data["imageWidth"] = w

                with open(f"{path4[:-3]}json", "w", encoding="utf-8") as f:
                    json.dump(file_data, f, ensure_ascii=False, indent="\t")
