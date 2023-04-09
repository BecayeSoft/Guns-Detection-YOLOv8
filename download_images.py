"""
A script to download images to be annoted for the dataset
"""

from ImageDownloader import ImageDownloader

queries = ['firearm', 'gun', 'gun shooting']
downloader = ImageDownloader()

# Download 'gun' and 'gun shooting' images
for query in queries:
    downloader.simple_image_download(query=query)
    downloader.unsplash(query=query)
    downloader.pixabay(query=query)


# Download 'Guns in public' images (from Google only)
# Pixabay and Unsplash did not give pertinent results on that.
downloader.simple_image_download(query='gun in  public')

