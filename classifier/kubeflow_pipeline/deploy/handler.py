from MyHandler import MyHandler

_service = MyHandler()

def handle(data, context):
    if not _service.initialized:
        _service.initialized(context)

        if data in None:
            return None

        data = _service.preprocess(data)
        data = _service.inference(data)
        data = _service.postprocess(data)

        return data
