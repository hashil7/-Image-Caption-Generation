
import pickle
import numpy as np

from keras.preprocessing.image import load_img, img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import VGG16

from keras.models import Model
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from PIL import Image



from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from io import BytesIO

model_cnn = VGG16()
model_cnn = Model(inputs = model_cnn.inputs , outputs = model_cnn.layers[-2].output)
model = load_model(r'E:\Users\My PC\Documents\DL-Projects\4000_image_flicker_4_5_2024.h5')
max_length = 35
with open(r'E:\Users\My PC\Documents\DL-Projects\tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

app = FastAPI()
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials = True,
    allow_methods=["*"],
    allow_headers=['*'],
)

@app.get("/")
async def health_check():
    return "The health check is successful"


@app.post("/generate-caption/")
async def predict_image(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(BytesIO(contents))
    caption =generate_caption(image)
    words = caption.split()
    modified_caption = ' '.join(words[1:-1]) if len(words) > 2 else ''
    return {"caption": modified_caption}
    

def idx_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None
# generate caption for an image
def predict_caption(model, image, tokenizer, max_length):
    # add start tag for generation process
    in_text = 'startseq'
    # iterate over the max length of sequence
    for i in range(max_length):
        # encode input sequence
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        # pad the sequence
        sequence = pad_sequences([sequence], max_length)
        # predict next word
        yhat = model.predict([image, sequence], verbose=0)
        # get index with high probability
        yhat = np.argmax(yhat)
        # convert index to word
        word = idx_to_word(yhat, tokenizer)
        # stop if word not found
        if word is None:
            break
        # append word as input for generating next word
        in_text += " " + word
        # stop if we reach end tag
        if word == 'endseq':
            break
    return in_text



def get_feature(image):
    image = image.resize((224,224))
    
    image = img_to_array(image)

    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    
    image = preprocess_input(image)
    
    feature = model_cnn.predict(image, verbose=0)
    return feature

def generate_caption(image):
    feature = get_feature(image)
    y_pred = predict_caption(model, feature, tokenizer, max_length)
    
    print(y_pred)
    return y_pred

