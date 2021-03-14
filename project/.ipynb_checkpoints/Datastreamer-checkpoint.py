# streamer class to fill word2vec list by list
class DataStreamer():
    def __init__(self, data):
        self.data = data

    def __iter__(self):
        for basket in self.data:
            yield basket.tolist()