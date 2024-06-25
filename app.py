import io
from google.cloud import vision
from google.oauth2 import service_account
from elasticsearch import Elasticsearch, helpers
import openai
import gradio as gr
from datetime import datetime
import environ
import os

env = environ.Env(
    # set casting, default value
    DEBUG=(bool, False)
)

# Set the project base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Take environment variables from .env file
environ.Env.read_env(os.path.join(BASE_DIR, '.env'))

# False if not in os.environ because of casting above
DEBUG = env('DEBUG')

# Initialize Google Cloud Vision client
# credentials = service_account.Credentials.from_service_account_file('path/to/your/service_account_key.json')
credentials = service_account.Credentials.from_service_account_file(env('CLOUD_VISION_KEY'))
vision_client = vision.ImageAnnotatorClient(credentials=credentials)

# Initialize Elasticsearch client
es = Elasticsearch([{'host': 'localhost', 'port': 9200}])

# Set OpenAI API key
# openai.api_key = 'your-openai-api-key'
openai.api_key = env('OPENAPI_KEY')

def analyze_image(image_file_path):
    with io.open(image_file_path, 'rb') as image_file:
        content = image_file.read()

    image = vision.Image(content=content)
    response = vision_client.annotate_image({
        'image': image,
        'features': [
            {'type': vision.Feature.Type.TEXT_DETECTION},
            {'type': vision.Feature.Type.FACE_DETECTION},
            {'type': vision.Feature.Type.OBJECT_LOCALIZATION}
        ],
    })

    extracted_text = ""
    faces = []
    objects = []

    if response.text_annotations:
        extracted_text = response.text_annotations[0].description

    if response.face_annotations:
        for face in response.face_annotations:
            face_data = {
                'anger': face.anger_likelihood,
                'joy': face.joy_likelihood,
                'sorrow': face.sorrow_likelihood,
                'surprise': face.surprise_likelihood,
                'detection_confidence': face.detection_confidence,
                'bounding_poly': [(vertex.x, vertex.y) for vertex in face.bounding_poly.vertices]
            }
            faces.append(face_data)

    if response.localized_object_annotations:
        for obj in response.localized_object_annotations:
            objects.append(obj.name)

    return extracted_text, faces, objects

def search_images(query):
    body = {
        "query": {
            "multi_match": {
                "query": query,
                "fields": ["description", "extracted_text", "faces.attributes", "objects.name"]
            }
        }
    }
    results = es.search(index='personal_images', body=body)
    return results['hits']['hits']

# Dummy search functions for other personal data
def search_resume(query): return []
def search_thesis(query): return []
def search_photos_events(query): return []
def search_daily_conversations(query): return []
def search_calendar_events(query): return []

def process_input(input_text=None, input_image=None):
    context = ""

    if input_text:
        resume_results = search_resume(input_text)
        thesis_results = search_thesis(input_text)
        photos_events_results = search_photos_events(input_text)
        daily_conversations_results = search_daily_conversations(input_text)
        calendar_events_results = search_calendar_events(input_text)
        image_results = search_images(input_text)

        combined_results = (
            [res['_source'] for res in resume_results] +
            [res['_source'] for res in thesis_results] +
            [res['_source'] for res in photos_events_results] +
            [res['_source'] for res in daily_conversations_results] +
            [res['_source'] for res in calendar_events_results] +
            [res['_source'] for res in image_results]
        )
        
        context += "\n".join([str(result) for result in combined_results])

    if input_image:
        extracted_text, faces, objects = analyze_image(input_image)
        context += f"\nExtracted Text from Image: {extracted_text}"
        context += f"\nDetected Faces: {', '.join([str(face) for face in faces])}"
        context += f"\nDetected Objects: {', '.join(objects)}"

        if extracted_text:
            resume_results = search_resume(extracted_text)
            thesis_results = search_thesis(extracted_text)
            photos_events_results = search_photos_events(extracted_text)
            daily_conversations_results = search_daily_conversations(extracted_text)
            calendar_events_results = search_calendar_events(extracted_text)
            image_results = search_images(extracted_text)

            combined_results = (
                [res['_source'] for res in resume_results] +
                [res['_source'] for res in thesis_results] +
                [res['_source'] for res in photos_events_results] +
                [res['_source'] for res in daily_conversations_results] +
                [res['_source'] for res in calendar_events_results] +
                [res['_source'] for res in image_results]
            )
            
            context += "\n".join([str(result) for result in combined_results])

    combined_prompt = context + "\n" + (input_text or "")
    response = openai.Completion.create(
        engine="text-davinci-003",  # Use text-davinci-003 for GPT-3.5
        prompt=combined_prompt,
        max_tokens=150,
        temperature=0.7,
    )
    response_text = response.choices[0].text.strip()

    return response_text

# Create Gradio interface
iface = gr.Interface(
    fn=process_input,
    inputs=[gr.Textbox(label="Text Prompt"), gr.Image(source="upload", type="filepath", label="Image Prompt")],
    outputs="text",
    title="Personalized Chat AI with Elasticsearch and Image Recognition"
)
iface.launch()
