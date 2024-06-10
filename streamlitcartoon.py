import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

def cartoonize_image(img):
    # Convert image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Apply median blur
    gray = cv2.medianBlur(gray, 3)
    
    # Detect edges using adaptive thresholding and edge-preserving filter
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                  cv2.THRESH_BINARY, 7, 7)
    
    # Apply bilateral filter to smoothen the image
    color = cv2.bilateralFilter(img, 9, 300, 300)
    
    # Combine edges and smoothed color image
    cartoon = cv2.bitwise_and(color, color, mask=edges)
    
    # Enhance cartoon effect by applying stylization
    cartoon = cv2.stylization(cartoon, sigma_s=150, sigma_r=0.25)
    
    return cartoon

def main():
    st.title("Image Cartoonizer App")
    st.write("Upload an image to convert it to a cartoon.")

    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        try:
            # To read the image file buffer with OpenCV
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            if img is not None:
                # Display the uploaded image
                st.image(img, channels="BGR", caption="Uploaded Image")

                # Get the name of the uploaded file
                st.write(f"File name: {uploaded_file.name}")

                # Convert the image to a cartoon
                cartoon_img = cartoonize_image(img)
                # Display the cartoon image
                st.image(cartoon_img, channels="BGR", caption="Cartoonized Image")

                # Convert the cartoon image to a format that can be downloaded
                is_success, buffer = cv2.imencode(".png", cartoon_img)
                if is_success:
                    byte_io = io.BytesIO(buffer)
                    # Download button
                    st.download_button(
                        label="Download Cartoonized Image",
                        data=byte_io.getvalue(),
                        file_name="cartoonized_image.png",
                        mime="image/png"
                    )
                else:
                    st.error("Error: The cartoonized image could not be converted for download.")
            else:
                st.error("Error: The uploaded file could not be processed as an image.")
        except Exception as e:
            st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
