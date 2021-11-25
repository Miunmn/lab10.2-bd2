from rtree import index
import face_recognition
from pathlib import Path
import os 
import numpy as np
import random
import matplotlib.pyplot as plt


def get_encodings(Rtree):
    i = 0
    final_encodings=[]
    path = "lfw-a/lfw/"
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
                        Rtree.insert(i, listAux, ( path +  str(name) + '/' + str(item.name) ))
                        final_encodings.append(listAux)
                        i= i+ 1
    Rtree.close()
    return final_encodings  

def generate_distances():
    distances = []
    people = []
    encodings = []

    for root, folder, files in os.walk('lfw-a/'):
        if root == '../input/alldata/Project/data/lfw/':
            continue
        for item in files:
            print(item)
            people.append(os.path.join(root, item))

    N = 500
    
    for i in range (N):
        print("Iteration", i)
        random_person_1 = random.sample(people, 1)[0]
        random_person_2 = random.sample(people, 1)[0]

        image_person_1 = face_recognition.load_image_file(random_person_1)
        image_person_2 = face_recognition.load_image_file(random_person_2)

        face_encoding_1 = face_recognition.face_encodings(image_person_1)
        face_encoding_2 = face_recognition.face_encodings(image_person_2)
        
        face_encoding_1 = np.array(face_encoding_1)
        face_encoding_2 = np.array(face_encoding_2)
        
        try:
            distances.append(np.linalg.norm(face_encoding_1 - face_encoding_2))
        except:
            print("Failed images", random_person_1, random_person_2)

    return distances

if __name__ == "__main__":
    distances = generate_distances()
    plt.hist(distances, 10)
    plt.show()
