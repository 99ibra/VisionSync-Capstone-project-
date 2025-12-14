import sys
from gradio_client import Client
import pyttsx3
import cv2

# Initialize the TTS engine
engine = pyttsx3.init()

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # ,cv2.CAP_DSHOW speeds up the camera initialization by a lot

while cap.isOpened():
    success, img = cap.read()

    k = cv2.waitKey(5)

    if k == 27:
        break
    elif k == ord('s'):  # wait for 's' key to save and exit
        cv2.imwrite('imageFolder/tempImage' + '.png', img)
        print("Image saved!")
        cap.release()  # Release the camera capture object after saving the image
        cv2.destroyAllWindows()  # Close the OpenCV window
        

    cv2.imshow('Image', img)

client = Client("https://ritaparadaramos-smallcapdemo.hf.space/")
result = client.predict(
    "imageFolder/tempImage.png",  # str (filepath or URL to image) in 'image' Image component
    api_name="/predict"
)

# Print the result
print(result)

# Extract the first sentence before "Retrieved captions:"
first_sentence = result.split("Retrieved captions:")[0].strip()

# Use the TTS engine to read the first sentence aloud
engine.say(first_sentence)
engine.runAndWait()

# Exit the program after processing is complete
sys.exit()
