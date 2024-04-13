from duckduckgo_search import DDGS

class search:
    def __init__(self):
        self.ddgs = DDGS()

    def search_videos(self, keywords, region="us-en", safesearch="off", timelimit=None, resolution=None, duration=None, max_results=4):
        results = self.ddgs.videos(
            keywords=keywords,
            region=region,
            safesearch=safesearch,
            timelimit=timelimit,
            resolution=resolution,
            duration=duration,
            max_results=max_results,
        )
        video_embeds = []
        for result in results:
            if result['embed_url'] != '':
                video_embeds.append(f"<embed src='{result['embed_url']}' width='560' height='315'>")
        return video_embeds
    def search_text(self, query, max_results=5, top=5):
        results = self.ddgs.text(query, max_results=max_results, region="us-en")
        full_list = []
        final_text = ""
        texts = ""
        urls = []
        for result in results:
            url = result['href']
            urls.append(url)
            full_list.append(result['body'])

        for i, txt in enumerate(full_list):
            final_text += f"This text is from this url {urls[i]}"
            final_text += "\n"
            final_text += txt
            final_text += "\n"
        return final_text
    def search_images(self, keywords, region="us-en", safesearch="off", size=None, color=None, type_image=None, layout=None, license_image=None, max_results=4):
        images = []
        results = self.ddgs.images(
            keywords=keywords,
            region=region,
            safesearch=safesearch,
            size=size,
            color=color,
            type_image=type_image,
            layout=layout,
            license_image=license_image,
            max_results=max_results
        )
        for img in results:
            images.append((f'<img src="{img["image"]}">'))   
        return images
