import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

image = Image.open("jonathan.jpeg")
width = image.size[0]
length = image.size[1]
jonathancolor = np.array(image.getdata()) # color data for K means
jonathan = jonathancolor.reshape(length, width, 3)

# Calculate how many pixels are in the image
n_pixels = width * length
print(f"Number of pixels in the image: {n_pixels}")

# Use scikit-learn to cluster the colors in the image 
# for k = 4, 8, 16. It is recommended to use n_jobs=-1 
# to take advantage of parallel processing when running K-means. 
# Save the centers of each cluster, remembering to convert the centers back to integers from float.
k_values = [4, 8, 16]
clustered_images = []

for k in k_values:
    kmeans = KMeans(n_clusters=k)
    # Fit K-means on the color data
    kmeans.fit(jonathancolor)
    # Get the cluster centers and convert to integers
    centers = np.array(kmeans.cluster_centers_, dtype=np.uint8)
    print(f"Cluster centers for k={k}: {centers}")
    # Get labels for all points
    labels = kmeans.labels_
    # Replace each pixel with its corresponding cluster center
    clustered_image_data = centers[labels]
    clustered_image = clustered_image_data.reshape(length, width, 3)
    clustered_images.append(clustered_image)


for i, clustered_image in enumerate(clustered_images):
    plt.figure(i+1)
    plt.clf()
    plt.axis('off')
    plt.title(f'Compressed image (k={k_values[i]} colors)')
    plt.imshow(clustered_image)
    plt.show()
