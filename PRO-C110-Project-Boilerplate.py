# To Capture Frame
import cv2

# To process image array
import numpy as np

# import the tensorflow modules and load the model
import tensorflow as tf # type: ignore

# Load the pre-trained model (assuming you have a model saved as 'model.h5')
model = tf.keras.models.load_model('model.h5')

# Attaching Cam indexed as 0, with the application software
camera = cv2.VideoCapture(0)

# Infinite loop
while True:
    # Reading / Requesting a Frame from the Camera 
    status, frame = camera.read()

    # if we were successfully able to read the frame
    if status:
        # Flip the frame
        frame = cv2.flip(frame, 1)
        
        # Resize the frame to the size the model expects (assuming 224x224)
        resized_frame = cv2.resize(frame, (224, 224))
        
        # Expand the dimensions to match the model's input shape (1, 224, 224, 3)
        expanded_frame = np.expand_dims(resized_frame, axis=0)
        
        # Normalize the image (assuming the model expects values in range 0-1)
        normalized_frame = expanded_frame / 255.0
        
        # Get predictions from the model
        predictions = model.predict(normalized_frame)
        
        # Display the predictions on the frame (optional, assuming it's a classification model)
        cv2.putText(frame, f'Prediction: {np.argmax(predictions)}', (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        
        # Displaying the frames captured
        cv2.imshow('feed', frame)

        # Waiting for 1ms
        code = cv2.waitKey(1)
        
        # If space key is pressed, break the loop
        if code == 32:
            break

# Release the camera from the application software
camera.release()

# Close the open window
cv2.destroyAllWindows()