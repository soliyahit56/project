# import streamlit as st
# import cv2
# import numpy as np
# from sklearn.cluster import KMeans
# from PIL import Image

# # Set up Streamlit title and description
# st.title('Image Processing with K-Means and Edge Detection')
# st.write('Upload an image to perform edge detection and color reduction using K-Means clustering.')

# # Upload image section
# uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# if uploaded_file is not None:
#     # Read image
#     img = Image.open(uploaded_file)
#     img = np.array(img)
#     img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

#     # Display the original image
#     img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert back to RGB for correct display
#     st.image(img_rgb, caption='Original Image', use_column_width=True)

#     # Perform edge detection
#     line_size = 7
#     blur_value = 7

#     gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     gray_blur = cv2.medianBlur(gray_img, blur_value)
#     edges = cv2.adaptiveThreshold(gray_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, line_size, blur_value)

#     # st.image(edges, caption='Edge Detection', use_column_width=True, channels='GRAY')

#     # Perform K-Means color clustering
#     k = 7
#     data = img.reshape(-1, 3)

#     kmeans = KMeans(n_clusters=k, random_state=42).fit(data)
#     img_reduced = kmeans.cluster_centers_[kmeans.labels_]
#     img_reduced = img_reduced.reshape(img.shape)
#     img_reduced = img_reduced.astype(np.uint8)

#     img_reduced_rgb = cv2.cvtColor(img_reduced, cv2.COLOR_BGR2RGB)  # Convert back to RGB for correct display
#     st.image(img_reduced_rgb, caption='Color Reduced Image (K-Means)', use_column_width=True)

#     cartoon_image_bytes = convert_to_bytes(img_reduced_rgb)
#     st.download_button(
#         label="Download Cartoonized Image",
#         data=cartoon_image_bytes,
#         file_name="cartoonized_image.png",
#         mime="image/png"
#     )

# # To run the app, save this as `app.py` and use the command `streamlit run app.py` in your terminal.
# #.\myenv\Scripts\activate 

import streamlit as st
import cv2
import numpy as np
from sklearn.cluster import KMeans
from PIL import Image

def main():
    # Streamlit Title and Description
    st.title("Image Processing with K-Means and Edge Detection")
    st.write("Upload an image to apply edge detection and color reduction using K-Means clustering.")

    # File Uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Load and Display Original Image
        img = Image.open(uploaded_file)
        img = np.array(img)
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        st.image(img_rgb, caption="Original Image", use_column_width=True)

        # Edge Detection
        edges = apply_edge_detection(img_bgr)
        # st.image(edges, caption="Edge Detection", use_column_width=True, channels="GRAY")

        # Color Reduction using K-Means
        k = 7
        img_reduced = apply_kmeans_clustering(img_bgr, k)
        img_reduced_rgb = cv2.cvtColor(img_reduced, cv2.COLOR_BGR2RGB)
        st.image(img_reduced_rgb, caption="Color Reduced Image (K-Means)", use_column_width=True)

        # Download Cartoonized Image
        cartoon_image_bytes = convert_to_bytes(img_reduced_rgb)
        st.download_button(
            label="Download Cartoonized Image",
            data=cartoon_image_bytes,
            file_name="cartoonized_image.png",
            mime="image/png"
        )

def apply_edge_detection(img):
    """Applies edge detection to the given image."""
    line_size = 7
    blur_value = 7

    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.medianBlur(gray_img, blur_value)
    edges = cv2.adaptiveThreshold(
        gray_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, line_size, blur_value
    )
    return edges

def apply_kmeans_clustering(img, k):
    """Reduces the color palette of the image using K-Means clustering."""
    data = img.reshape(-1, 3)
    kmeans = KMeans(n_clusters=k, random_state=42).fit(data)
    img_reduced = kmeans.cluster_centers_[kmeans.labels_]
    img_reduced = img_reduced.reshape(img.shape).astype(np.uint8)
    return img_reduced

def convert_to_bytes(img):
    """Converts an image array to bytes for download."""
    is_success, buffer = cv2.imencode(".png", img)
    return buffer.tobytes()

if __name__ == "__main__":
    main()
