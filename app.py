import streamlit as st
import cv2
import numpy as np
from io import BytesIO

# Function to cartoonify an image
def cartoonify_image(image):
    # Convert the image to an OpenCV format
    img = np.array(image)
    img = img[:, :, ::-1].copy()  # Convert RGB to BGR
    
    # Step 1: Resize the image for better performance (optional)
    img = cv2.resize(img, (800, 600))

    # Step 2: Convert the image to gray scale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Step 3: Apply a median blur to smooth the image
    gray = cv2.medianBlur(gray, 5)

    # Step 4: Detect edges using adaptive thresholding
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                  cv2.THRESH_BINARY, 9, 9)

    # Step 5: Apply bilateral filter for smoothening
    color = cv2.bilateralFilter(img, 9, 300, 300)

    # Step 6: Combine the edges and the smoothed image
    cartoon = cv2.bitwise_and(color, color, mask=edges)

    return cartoon

# Streamlit app UI
st.title("Cartoonify Your Image")

st.write("Upload an image and see it turned into a cartoon!")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    # Read the image file
    image = st.image(uploaded_file)

    # Convert the uploaded image to a format suitable for cartoonification
    img_bytes = uploaded_file.read()
    img = np.array(bytearray(img_bytes), dtype=np.uint8)
    img = cv2.imdecode(img, 1)

    # Call the cartoonify function
    cartoon_image = cartoonify_image(img)

    # Display the cartoonified image
    st.image(cartoon_image, caption="Cartoonified Image", use_column_width=True)

    # Save the cartoon image if needed
    output_path = "cartoonified_image.jpg"
    cv2.imwrite(output_path, cartoon_image)
    st.write(f"Cartoonified image saved as {output_path}")
