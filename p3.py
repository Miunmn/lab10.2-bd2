import face_recognition
from rtree import index 
from pathlib import Path
import numpy as np
import heapq

def knn_search_sequential_pq(k, Q, n):  
    path = "lfw-a\lfw\\"
    basepath = Path(path)
    aux = 0  
    route = []
    faces = []
    for entry in basepath.iterdir():
        if entry.is_dir():
            name = entry.name
            files = Path(path + str(name))
            files_in_basepath = files.iterdir()
            for item in files_in_basepath:
                if item.is_file():
                    image_name = path + str(name) + item.name
                    image = face_recognition.load_image_file(item)
                    facesEncoding = face_recognition.face_encodings(image)                    
                    for faceEncoding in facesEncoding:
                        if aux == n:
                            dist = face_recognition.face_distance(faces, Q)
                            auxArr = []
                            for i in range(0, aux, 1):
                                auxArr.append((dist[i], route[i]))
                            heapq.heapify(auxArr)    
                            return heapq.nsmallest(k, auxArr)
                        route.append(image_name)
                        faces.append(faceEncoding)        
                        aux += 1
    dist = face_recognition.face_distance(faces, Q)
    auxArr = [] 

    for i in range(0, aux, 1):
        auxArr.append((dist[i], route[i]))
    heapq.heapify(auxArr)    
    return heapq.nsmallest(k, auxArr)

def knn_search_rtree(k, Q):
    p = index.Property()
    p.dimension = 128 #D
    p.buffering_capacity = 10 #M
    Rtree = index.Rtree("RtreeKnn", properties=p)
    coordinatesListQuery = list(Q)
    for i in Q:
        coordinatesListQuery.append(i)

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
    return list(Rtree.nearest(coordinates=coordinatesListQuery ,num_results=k, objects='raw'))



def bounding_box(v: np.ndarray, r: float) -> np.ndarray:
    return np.concatenate((v-(r/2), v+(r/2)), axis=None)

def range_search_rtree(query, radio):
    p = index.Property()
    p.dimension = 128
    p.buffering_capacity = 10
    Rtree = index.Index('RtreeRangeSearch', properties=p)
    return [
        n.object for n in Rtree.intersection(bounding_box(query, radio), objects=True)
        if n is not None and isinstance(n.object, str)
    ]


if __name__ == "__main__":
    print("knn_search_sequential_pq: ")
    print(knn_search_sequential_pq(3, face_recognition.face_encodings(face_recognition.load_image_file('lfw-a\lfw\Aaron_Pena\Aaron_Pena_0001.jpg'))[0], 50))
    print("knn_search_rtree: ")
    print(knn_search_rtree(3, face_recognition.face_encodings(face_recognition.load_image_file('lfw-a\lfw\Aaron_Pena\Aaron_Pena_0001.jpg'))[0]))
    print("range_search_rtree: ")
    print(range_search_rtree(face_recognition.face_encodings(face_recognition.load_image_file('lfw-a/lfw/Aaron_Pena/Aaron_Pena_0001.jpg'))[0], 3.0 ))
