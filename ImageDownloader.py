import os
import shutil
import requests
import json
from pixabay import image
from simple_image_download import simple_image_download as sid


class ImageDownloader:
    """
    A class that contains methods to download images from Google, unsplash and pixabay
    using a simple `query` that contanins keywords.
    """

    def __init__(self, query=None):
        """
        Create an ImageDownloder

        :param query: a string that contains keyword(s) to search and download the images.
        """

        self.pixabay_key = '15625626-1238828730abaf9c134a2238a'
        self.unsplash_key = 'm8VNiQK6r3ETdHG9rbhvbd81O1ERoYYF4zSveI8tvlY'
        self.query = query
        self.download_dir = "downloads"
        # self.make_dirs()

    def make_dirs(self, query=None):
        """
        Create a directory to store the downloaded images.
        The directory name is the same as the query and is placed inside `downloads`.
        E.g: 'downloads/guns'.

        :param query: a string that contains keyword(s) to search and download the images.
        It will be used as the directory name.
        If not specified, `self.query` will be used.
        """

        if query is None:
            query = self.query

        if not os.path.exists(self.download_dir):
            os.makedirs(self.download_dir)

        query_dir = os.path.join(self.download_dir, query)
        if not os.path.exists(query_dir):
            os.makedirs(query_dir)

    def unsplash(self, query=None):
        """
        Download images from unsplash.

        :param query: a string that contains keyword(s) to search and download the images.
        If not specified, `self.query` will be used.
        """

        if query is None:
            query = self.query

        # create folder to store the images
        self.make_dirs(query)

        # Define search query and API endpoint
        endpoint = "https://api.unsplash.com/search/photos"

        # Set up parameters for first request
        params = {
            "query": query,
            "per_page": 20,
            "page": 1,
            "client_id": self.unsplash_key
        }

        # Send first request and get total number of pages
        response = requests.get(endpoint, params=params)
        data = json.loads(response.text)
        total_pages = data["total_pages"]

        # Download through each page
        for page in range(1, total_pages):
            params["page"] = page
            response = requests.get(endpoint, params=params)
            data = json.loads(response.text)

            for result in data["results"]:
                image_url = result["urls"]["regular"]
                image_id = result["id"]
                response = requests.get(image_url)

                with open(f"{self.download_dir}/{self.query}/{image_id}.jpg", "wb") as f:
                    f.write(response.content)
                    print(f"Downloaded {image_id}.jpg")

    def pixabay(self, query=None):
        """
        Download images from pixabay

        :param query: a string that contains keyword(s) to search and download the images.
        If not specified, `self.query` will be used.
        """

        if query is None:
            query = self.query

        # create folder to store the images
        self.make_dirs(query)

        image = image(self.pixabay_key)
        results = image.search(q=query,
                               lang='en',
                               image_type='photo',
                               per_page=10)

        # Download the images
        for i, result in enumerate(results['hits']):
            image_url = result['largeImageURL']
            image_id = result['id']
            filename = os.path.join(self.download_dir, query, f"{image_id}.jpg")
            image.download(image_url, filename)

    def simple_image_download(self, query=None, limit=100):
        """
        Download images using `simple_image_downloader`

        :param query: a string that contains keyword(s) to search and download the images.
        If not specified, `self.query` will be used.
        :param limit: the number of images to be downloaded
        """

        if query is None:
            query = self.query

        self.make_dirs(query)

        response = sid.Downloader()
        response.download(query, limit=limit)

        # move query folder to the downloads directory
        source_dir = f"simple_images/{query}"
        dest_dir = f"{self.download_dir}/"
        shutil.move(source_dir, dest_dir)
