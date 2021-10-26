import numpy as np
from PIL import Image
import os
import csv
from sklearn.cluster import KMeans
from sklearn.metrics import euclidean_distances
from skimage.color import rgb2lab, lab2rgb
import matplotlib.pyplot as plt
import ClusterSegmentation as cls

# definere hovedmappe
mainFolder = os.getcwd()
filNavn = "reslt2.csv"


# Laver csv fil
with open(f'{filNavn}', 'a'):
    pass

# Undersøger om der er tidligere data
with open(f'{filNavn}', 'r') as file:
    reader = csv.reader(file)

    # Sætter start til sidste række
    start = len(list(reader))


def cluster_assignments(X, Y):
    return np.argmin(euclidean_distances(X, Y), axis=1)


# Pixeldata er et 2D array i formen (z,y,3) -> (z*y,3)
def centroidFinder(iterations, n_clusters, pixelData):
    K = n_clusters

    centers = np.array([pixelData.mean(0) + (np.random.randn(3) / 10) for _ in range(K)])

    y_kmeans = cluster_assignments(pixelData, centers)
    iterations = 30
    for i in range(iterations):
        # assign each point to the closest center
        y_kmeans = cluster_assignments(pixelData, centers)

        # move the centers to the mean of their assigned points (if any)
        for i, c in enumerate(centers):
            points = pixelData[y_kmeans == i]
            if len(points):
                centers[i] = points.mean(0)
    centers = np.array(centers)
    return (centers)




def guessStatus(nearestClusterIndex, rottenIndex, freshIndex, percentageRotten):
    rotten = 0
    fresh = 0
    for i in rottenIndex:
        rotten = rotten + np.count_nonzero(nearestClusterIndex == i)
    for j in freshIndex:
        fresh = fresh + np.count_nonzero(nearestClusterIndex == j)
    if fresh == 0:
        return (1)
    elif (rotten / fresh * 100 >= percentageRotten):
        return (1)
    else:
        return (0)


# funktion der skriver resultater til CSV fil
def writeToCsv(filNavn,index, rottenGuess, rottenReal):
    # bedømmer om algoritmen gætter rigtigt
    if rottenGuess == rottenReal:
        correct = 1
    else:
        correct = 0

    # Definerer hvad der skal skrives i CSV filen
    row = [index, rottenGuess, rottenReal, correct]

    # Sikrer at vi arbejder i hovedmappen
    os.chdir(mainFolder)

    # skriver resultater i CSV filen "results"
    with open(f'{filNavn}', 'a', newline='') as f:
        write = csv.writer(f)
        write.writerow(row)


# Returner en 1, hvis Image indeholder en rådden banan, returnerer 0, hvis den indeholder en frisk/ikke rådden banan.
def guessImage(image, n_clusters,iterations, notRottenMin, notRottenMax, rottenMin, rottenMax, notRottenGreyscale, rottenGreyscale,
               maximumInternalDifference, percentageRotten):
    # Omdanner billede til array
    originalImage = Image.open(image).convert("RGB")
    originalImage = rgb2lab(originalImage)
    shapeOriginalImage = np.shape(originalImage)  # Bruges til at reshape det 2D billede array
    imageArray = np.asarray(originalImage)

    twoDImageArray = imageArray.reshape(shapeOriginalImage[0] * shapeOriginalImage[1], 3)  # Laver (z,y,3) -> (z*y,3)
    # centroids = RosshawksKMean(datainput)
    #kMeans = KMeans(n_clusters=n_clusters, random_state=0).fit(twoDImageArray)  # Dataen fra n-means, med 300 iterations
    #centroids = kMeans.cluster_centers_  # Et to-dimensionelt array med R^3 vektorer.
    centroids = centroidFinder(iterations,n_clusters,twoDImageArray)


    nearestCentroidIndex = np.argmin(euclidean_distances(twoDImageArray, centroids), axis=1)  # 1D List with the centroid each datapoint belongs to

    centChecker = cls.centroidStatusChecker(centroids, notRottenMin, notRottenMax, notRottenGreyscale, rottenMin,
                                            rottenMax, rottenGreyscale, maximumInternalDifference)
    indexRottenCentroids = np.array(centChecker.indexRottenClusters)
    indexFreshCentroids = np.array(centChecker.indexNotRottenClusters)

    return (guessStatus(nearestCentroidIndex, rottenIndex=indexRottenCentroids, freshIndex=indexFreshCentroids,
                        percentageRotten=percentageRotten))


"""Hovedfunktionen:
    
    folderNames: En liste med navnene på de folders filerne ligger i.
    rottenStatus: En tilsvarende liste, hvor folder[i] har rottenstatus[i].
    n_clusters: Antallet af clusters til vores kmeans-segmentation
    notRotten- min/max: Ligger clusterens RGB-skala i mellem disse to, antager vi, at den indikerer en frisk frugt
    rotten- min/max: Ligger clusterens RGB-skala i mellem disse to, antager vi, at den indikerer en rådden frugt
    
    Bemærk her: At vi her tjekker hvert individuelt element i RGB-skalaen, matematisk:
    notRottenMin[j] <_ centroids[i][j] <_ notRottenMax[j]
    Hvor j = [0;3]
    i = antallet af centroids
     
    greyScale: RGB værdierne skal være lige store, før de kan vurderes som friske eller råddne.
    Bemærk her: Hvis eks. notRottenMin[i] != notRottenMin[j], så vil notRottenGreyscale = True automatisk udelukke alle friske clusters
    
    Hvor j,i = [0;3] og j =! i
"""


def imageSegmentation(resultFileName,folderNames, rottenStatus):
    imageIndex = -1

    # Loop over mapperne
    for i in range(len(folderNames)):
        realStatus = rottenStatus[i]

        # Definere aktive mappe
        activeFolder = mainFolder + '\\' + folderNames[i]
        # Skifter aktiv mappe til aktiv mappe
        os.chdir(activeFolder)

        # Laver liste over indhold(billeder) i mappen
        images = os.listdir(os.getcwd())

        # Not rotten values
        notRottenMinRGB = [220, 220, 0]
        notRottenMaxRGB = [255, 255, 90]

        notRottenMinLAB = [85.2, -19.28, 84.51]
        notRottenMaxLAB = [97.43, -19.15, 75.50]

        # Rotten values
        rottenMinRGB = [0, 0, 0]
        rottenMaxRGB = [105, 105, 105]

        rottenMinLAB = [0, 0, 0]
        rottenMaxLAB = [40, 0, 0]

        # Loop over billeder
        for n in images:
            # Skifter aktiv mappe til aktiv mappe
            os.chdir(activeFolder)

            # Opdatere billedindex
            imageIndex += 1

            # Forsætter loop efter tidligere resultater 
            if start > imageIndex:
                continue

            guess = guessImage(n,5,30, notRottenMinLAB, notRottenMaxLAB, rottenMinLAB, rottenMaxLAB, False, False, 60, 20)

            # Kalder CSV funktion til arkivering af data
            writeToCsv(resultFileName,imageIndex, guess, realStatus)


# Kørsel af funktion


imageSegmentation(filNavn,['freshBanana', 'rottenBanana','freshBanana', 'rottenBanana'], [0, 1,0,1])
