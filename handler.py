from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

# 加载 Google 的病理模型
model_id = 'google/path-foundation'
path_pipeline = pipeline(task=Tasks.feature_extraction, model=model_id)

def handle_request(params):
    image_url = params.get("image_url")
    if not image_url:
        return {"error": "No image URL provided."}
    
    result = path_pipeline(image_url)
    return {"embedding": result["embedding"].tolist()}
