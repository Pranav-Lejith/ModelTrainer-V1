import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
import zipfile
from io import BytesIO
import time
import matplotlib.pyplot as plt

# Set page config
st.set_page_config(
    page_title="Creatus",
    page_icon='logo.png',
    menu_items={
        'About': "# :red[Creator]:blue[:] :violet[Pranav Lejith(:green[Amphibiar])]",
    },
    layout='wide',
    initial_sidebar_state='collapsed'  # Start with sidebar collapsed
)

# Function to hide sidebar
def hide_sidebar():
    st.markdown(
        """
        <style>
        .css-1544g2n.e1fqkh3o4 {
            display: none;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

# Function to show sidebar
def show_sidebar():
    st.markdown(
        """
        <style>
        .css-1544g2n.e1fqkh3o4 {
            display: block;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

# Initialize session state keys
if 'labels' not in st.session_state:
    st.session_state['labels'] = {}
if 'num_classes' not in st.session_state:
    st.session_state['num_classes'] = 0
if 'label_mapping' not in st.session_state:
    st.session_state['label_mapping'] = {}
if 'model' not in st.session_state:
    st.session_state['model'] = None
if 'is_developer' not in st.session_state:
    st.session_state['is_developer'] = False
if 'show_developer_splash' not in st.session_state:
    st.session_state['show_developer_splash'] = False
if 'initial_load' not in st.session_state:
    st.session_state['initial_load'] = True

# Developer authentication (hidden from normal users)
developer_commands = [
    'override protocol-amphibiar', 'override command-amphibiar', 
    'command override-amphibiar', 'command override-amphibiar23', 
    'control override-amphibiar', 'system override-amphibiar', 'user:amphibiar',
    'user:amphibiar-developer', 'user:amphibiar-admin', 'user:amphibiar-root',
    'control-admin', 'control-amphibiar','inititate override-amphibiar','currentuser:amphibiar',
    'initiate control override', 'initiate control','switch control'
]

# Custom HTML for splash screen with typewriter effect
def create_splash_html(text, color):
    return f"""
    <style>
    .typewriter h1 {{
      overflow: hidden;
      color: {color};
      white-space: nowrap;
      margin: 0 auto;
      letter-spacing: .15em;
      border-right: .15em solid orange;
      animation: typing 2s steps(30, end), blink-caret .5s step-end infinite;
    }}
    
    @keyframes typing {{
      from {{ width: 0 }}
      to {{ width: 100% }}
    }}
    
    @keyframes blink-caret {{
      from, to {{ border-color: transparent }}
      50% {{ border-color: orange }}
    }}
    </style>
    <div class="typewriter">
        <h1>{text}</h1>
    </div>
    """

# Main content
def main_content():
    st.title(":red[Creatus (Model Creator)]")

    # Sidebar for label input
    st.sidebar.title(":blue[Manage Labels]")

    label_input = st.sidebar.text_input("Enter a new label:")
    if st.sidebar.button("Add Label"):
        if label_input in developer_commands:
            st.session_state['is_developer'] = True
            st.session_state['show_developer_splash'] = True
            st.experimental_rerun()
        elif label_input and label_input not in st.session_state['labels']:
            st.session_state['labels'][label_input] = []
            st.session_state['num_classes'] += 1
            st.sidebar.success(f"Label '{label_input}' added!")
        else:
            st.sidebar.warning("Label already exists or is empty.")

    # Dropdown to select model export format
    export_format = st.sidebar.selectbox("Select model export format:", options=["tflite", "h5"])

    # Display the existing labels and allow image upload in rows
    if st.session_state['num_classes'] > 0:
        num_columns = 3  # Adjust this value for the number of columns you want
        cols = st.columns(num_columns)
        
        for i, label in enumerate(st.session_state['labels']):
            with cols[i % num_columns]:  # Wrap to the next line
                st.subheader(f"Upload images for label: {label}")
                uploaded_files = st.file_uploader(f"Upload images for {label}", accept_multiple_files=True, type=['jpg', 'jpeg', 'png'], key=label)
                
                if uploaded_files:
                    for uploaded_file in uploaded_files:
                        image_data = image.load_img(uploaded_file, target_size=(64, 64))
                        image_array = image.img_to_array(image_data)
                        st.session_state['labels'][label].append(image_array)
                    st.success(f"Uploaded {len(uploaded_files)} images for label '{label}'.")

    # Advanced options in sidebar
    with st.sidebar.expander("Advanced Options", expanded=st.session_state['is_developer']):
        epochs = st.number_input("Epochs", min_value=1, max_value=1000, value=10)
        learning_rate = st.number_input("Learning Rate", min_value=0.0001, max_value=0.1, value=0.001, format="%.4f")
        batch_size = st.number_input("Batch Size", min_value=1, max_value=128, value=32)
        
        # Define model_architecture with a default value
        model_architecture = "Simple CNN"
        
        if st.session_state['is_developer']:
            st.subheader("Developer Options")
            
            # Theme customization
            theme = st.selectbox("Theme", ["Light", "Dark", "Custom"])
            if theme == "Custom":
                primary_color = st.color_picker("Primary Color", "#FF4B4B")
                secondary_color = st.color_picker("Secondary Color", "#0068C9")
                background_color = st.color_picker("Background Color", "#FFFFFF")
                text_color = st.color_picker("Text Color", "#262730")
                
                # Apply custom theme
                st.markdown(f"""
                    <style>
                    :root {{
                        --primary-color: {primary_color};
                        --secondary-color: {secondary_color};
                        --background-color: {background_color};
                        --text-color: {text_color};
                    }}
                    body {{
                        color: var(--text-color);
                        background-color: var(--background-color);
                    }}
                    .stButton > button {{
                        color: var(--background-color);
                        background-color: var(--primary-color);
                    }}
                    .stTextInput > div > div > input {{
                        color: var(--text-color);
                    }}
                    </style>
                """, unsafe_allow_html=True)
            
            # Model architecture options
            model_architecture = st.selectbox("Model Architecture", ["Simple CNN", "VGG-like", "ResNet-like", "Custom"])
            if model_architecture == "Custom":
                num_conv_layers = st.number_input("Number of Convolutional Layers", min_value=1, max_value=10, value=3)
                num_dense_layers = st.number_input("Number of Dense Layers", min_value=1, max_value=5, value=2)
                activation_function = st.selectbox("Activation Function", ["relu", "leaky_relu", "elu", "selu"])
            
            # Data augmentation options
            data_augmentation = st.checkbox("Enable Data Augmentation")
            if data_augmentation:
                rotation_range = st.slider("Rotation Range", 0, 180, 20)
                zoom_range = st.slider("Zoom Range", 0.0, 1.0, 0.2)
                horizontal_flip = st.checkbox("Horizontal Flip", value=True)
                vertical_flip = st.checkbox("Vertical Flip")
            
            # Training options
            early_stopping = st.checkbox("Enable Early Stopping")
            if early_stopping:
                patience = st.number_input("Early Stopping Patience", min_value=1, max_value=20, value=5)
            
            # Optimization options
            optimizer = st.selectbox("Optimizer", ["Adam", "SGD", "RMSprop"])
            if optimizer == "SGD":
                momentum = st.slider("Momentum", 0.0, 1.0, 0.9)
            
            # Regularization options
            l2_regularization = st.checkbox("L2 Regularization")
            if l2_regularization:
                l2_lambda = st.number_input("L2 Lambda", min_value=0.0001, max_value=0.1, value=0.001, format="%.4f")
            
            dropout = st.checkbox("Dropout")
            if dropout:
                dropout_rate = st.slider("Dropout Rate", 0.0, 0.5, 0.2)
            
            # Advanced visualization options
            show_model_summary = st.checkbox("Show Model Summary")
            plot_training_history = st.checkbox("Plot Training History")
            
            # Export options
            export_tensorboard_logs = st.checkbox("Export TensorBoard Logs")

    # Button to train the model
    if st.session_state['num_classes'] > 1:
        if st.button("Train Model"):
            all_images = []
            all_labels = []
            st.session_state['label_mapping'] = {label: idx for idx, label in enumerate(st.session_state['labels'].keys())}
            
            for label, images in st.session_state['labels'].items():
                all_images.extend(images)
                all_labels.extend([st.session_state['label_mapping'][label]] * len(images))
            
            if len(all_images) > 0:
                st.write("Training the model...")
                progress_bar = st.progress(0)  # Initialize progress bar
                
                # Prepare training options
                training_options = {
                    "learning_rate": learning_rate,
                    "batch_size": batch_size,
                    "model_architecture": model_architecture,
                    "data_augmentation": st.session_state['is_developer'] and data_augmentation,
                    "early_stopping": st.session_state['is_developer'] and early_stopping,
                }
                
                if st.session_state['is_developer']:
                    if model_architecture == "Custom":
                        training_options.update({
                            "num_conv_layers": num_conv_layers,
                            "num_dense_layers": num_dense_layers,
                            "activation_function": activation_function,
                        })
                    
                    if data_augmentation:
                        training_options.update({
                            "rotation_range": rotation_range,
                            "zoom_range": zoom_range,
                            "horizontal_flip": horizontal_flip,
                            "vertical_flip": vertical_flip,
                        })
                    
                    if early_stopping:
                        training_options["patience"] = patience
                    
                    training_options["optimizer"] = optimizer
                    if optimizer == "SGD":
                        training_options["momentum"] = momentum
                    
                    if l2_regularization:
                        training_options["l2_lambda"] = l2_lambda
                    
                    if dropout:
                        training_options["dropout_rate"] = dropout_rate
                
                st.session_state['model'] = train_model(all_images, all_labels, st.session_state['num_classes'], epochs, progress_bar, **training_options)
                
                if st.session_state['is_developer']:
                    if show_model_summary:
                        st.subheader("Model Summary")
                        st.text(st.session_state['model'].summary())
                    
                    if plot_training_history and hasattr(st.session_state['model'], 'history'):
                        st.subheader("Training History")
                        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
                        ax1.plot(st.session_state['model'].history.history['accuracy'])
                        ax1.plot(st.session_state['model'].history.history['val_accuracy'])
                        ax1.set_title('Model Accuracy')
                        ax1.set_ylabel('Accuracy')
                        ax1.set_xlabel('Epoch')
                        ax1.legend(['Train', 'Validation'], loc='upper left')
                        
                        ax2.plot(st.session_state['model'].history.history['loss'])
                        ax2.plot(st.session_state['model'].history.history['val_loss'])
                        ax2.set_title('Model Loss')
                        ax2.set_ylabel('Loss')
                        ax2.set_xlabel('Epoch')
                        ax2.legend(['Train', 'Validation'], loc='upper left')
                        
                        st.pyplot(fig)
                    
                    if export_tensorboard_logs:
                        # Code to export TensorBoard logs
                        pass
                
                st.toast('Model Trained Successfully')
                st.success("Model trained!")
            else:
                st.error("Please upload some images before training.")
    else:
        st.warning("At least two labels are required to train the model.")

    # Option to test the trained model
    if st.session_state['model'] is not None:
        st.subheader("Test the trained model with a new image")
        test_image = st.file_uploader("Upload an image to test", type=['jpg', 'jpeg', 'png','webp'], key="test")
        
        if test_image:
            # Show image preview
            test_image_data = image.load_img(test_image, target_size=(64, 64))
            st.image(test_image_data, caption="Uploaded Image", use_column_width=True)

            test_image_array = image.img_to_array(test_image_data)
            predicted_label, confidence = test_model(st.session_state['model'], test_image_array, st.session_state['label_mapping'])

            st.write(f"Predicted Label: {predicted_label}")
            st.slider("Confidence Level (%)", min_value=1, max_value=100, value=int(confidence * 100), disabled=True)

    # Button to download the model
    if st.session_state['model'] is not None and st.button("Download Model"):
        try:
            predicted_label_code = ', '.join([f"'{label}'" for label in st.session_state['label_mapping']])
            
            if export_format == 'tflite':
                usage_code = f"""
    import tensorflow as tf
    import numpy as np

    # Load the model
    interpreter = tf.lite.Interpreter(model_path="model.tflite")
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Prepare the image (adjust this for your actual input)
    img = np.random.rand(1, 64, 64, 3).astype(np.float32)

    # Test the model
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()

    output = interpreter.get_tensor(output_details[0]['index'])
    predicted_label = np.argmax(output)
    predicted_label_code = [{predicted_label_code}]
    print(f"Predicted Label: {{predicted_label_code[predicted_label]}}")
    """
            elif export_format == 'h5':
                usage_code = f"""
    import tensorflow as tf

    # Load the model
    model = tf.keras.models.load_model('model.h5')

    # Prepare the image (adjust this for your actual input)
    img = np.random.rand(1, 64, 64, 3)

    # Test the model
    prediction = model.predict(img)
    predicted_label = np.argmax(prediction)
    predicted_label_code = [{predicted_label_code}]
    print(f"Predicted Label: {{predicted_label_code[predicted_label]}}")
    """
            
            buffer = save_model(st.session_state['model'], export_format, usage_code)
            
            st.download_button(
                label="Download the trained model and usage code",
                data=buffer,
                file_name=f"trained_model_{export_format}.zip",
                mime="application/zip"
            )
        except Exception as e:
            st.error(f"Error: {e}")

    st.sidebar.write("This app was created by :red[Pranav Lejith](:violet[Amphibiar])")
    st.sidebar.subheader(":orange[Usage Instructions]")
    st.sidebar.write(""" 
    1) Manage Labels: Enter a new label and upload images for that label.
                     
    2) Train Model: After uploading images for at least two labels, you can train the model.
                     
    3) Test Model: Once the model is trained, you can test it with new images and see predictions along with confidence levels.
                     
    4) Download Model: Finally, you can download the trained model in TensorFlow Lite or .h5 format for use in other applications. Tensorflow lite model is better because it is smaller in size as compared to the .h5 model so it can be used in many applications which have a file size limit.
                     

    """, unsafe_allow_html=True)
    st.sidebar.subheader(":red[Warning]")
    st.sidebar.write('The code might produce a ghosting effect sometimes. Do not panic due to the Ghosting effect. It is caused due to delay in code execution.')

    st.sidebar.subheader(":blue[Note]  :green[ from]  :red[ Developer]:")
    st.sidebar.write('The Creatus model creator is slightly more efficient than the teachable machine model creator as Creatus provides more customizability. But, for beginners, teachable machine might be a more comfortable option due to its simplicity and user friendly interface. But for advanced developers, Creatus will be more preferred choice.')
    st.sidebar.subheader(':blue[Definitions]  ')
    st.sidebar.write("""
    **:red[Batch Size]**: 
    Batch size is the number of samples that you feed into your model at each iteration of the training process. It determines how often you update the model parameters based on the gradient of the loss function. A larger batch size means more data per update, but also more memory and computation requirements.
    
    **:orange[Epochs]**:
    An epoch is when all the training data is used at once and is defined as the total number of iterations of all the training data in one cycle for training the machine learning model. Another way to define an epoch is the number of passes a training dataset takes around an algorithm
    
    **:violet[Learning Rate]**:
    Learning rate refers to the strength by which newly acquired information overrides old information. It determines how much importance is given to recent information compared to previous information during the learning process.
    """)
    # Add reset button for developer mode at the bottom of the sidebar
    if st.session_state['is_developer']:
        if st.sidebar.button("Reset to Normal User", key="reset_button"):
            st.session_state['is_developer'] = False
            st.experimental_rerun()

# Define a function to train the model with progress
def train_model(images, labels, num_classes, epochs, progress_bar, **kwargs):
    # ... (rest of the train_model function remains unchanged)
    pass

# Function to save the model in the specified format
def save_model(model, export_format, usage_code):
    # ... (rest of the save_model function remains unchanged)
    pass

# Function to test the model with a new image
def test_model(model, img_array, label_mapping):
    # ... (rest of the test_model function remains unchanged)
    pass

# Main app logic
if st.session_state['initial_load']:
    hide_sidebar()
    splash = st.empty()
    splash.markdown(create_splash_html("Creatus", '#48CFCB'), unsafe_allow_html=True)
    time.sleep(1)
    splash.empty()
    show_sidebar()
    st.session_state['initial_load'] = False
    main_content()
elif st.session_state['show_developer_splash']:
    hide_sidebar()
    # Create a container for the entire app content
    app_container = st.empty()
    
    # Show only the developer splash
    dev_splash = st.empty()
    dev_splash.markdown(create_splash_html("Welcome , Amphibiar (Developer)", 'red'), unsafe_allow_html=True)
    
    # Wait for the typing animation to complete (adjust the sleep time if needed)
    time.sleep(4)
    
    # Clear the developer splash
    dev_splash.empty()
    
    # Reset the developer splash flag
    st.session_state['show_developer_splash'] = False
    
    show_sidebar()
    
    # Show the main content
    with app_container.container():
        main_content()
else:
    main_content()
