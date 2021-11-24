import face_recognition
from rtree import index 
from pathlib import Path
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
    Rtree = index.Index('RtreeKnn', properties=p)
    coordinatesListQuery = list(Q)
    for i in Q:
        coordinatesListQuery.append(i)
    return list(Rtree.nearest(coordinatesListQuery, k, 'raw'))

if __name__ == "__main__":
    print(knn_search_sequential_pq(3,  face_recognition.face_encodings(face_recognition.load_image_file('lfw-a\lfw\Aaron_Pena\Aaron_Pena_0001.jpg'))[0], 50))
    print(knn_search_rtree(3,  face_recognition.face_encodings(face_recognition.load_image_file('lfw-a\lfw\Aaron_Pena\Aaron_Pena_0001.jpg'))[0]))