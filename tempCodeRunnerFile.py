from rtree import index
import face_recognition
from pathlib import Path

def get_encodings(Rtree):
    i = 0
    path = "lfw-a\lfw\\"
    basepath = Path(path)
    for entry in basepath.iterdir():
        if entry.is_dir():
            name = entry.name
            files = Path(path + str(name))
            files_in_basepath = files.iterdir()
            for item in files_in_basepath:
                if item.is_file():
                    image = face_recognition.load_image_file(item)
                    facesEncoding = face_recognition.face_encodings(image)
                    for faceEncoding in facesEncoding:
                        listAux = list(faceEncoding)
                        for cord in faceEncoding:
                            listAux.append(cord)
                        Rtree.insert(i, listAux, ( path +  str(name) + '\\' + str(item.name) ))
                        i= i+ 1
    Rtree.close()
    return Rtree  

if __name__ == "__main__":
    p = index.Property()
    p.dimension = 128 #D
    p.buffering_capacity = 10 #M
    Rtree = index.Index('RtreeLab', properties=p)
    encodings = get_encodings(Rtree)



