import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt

def make_gradcam_heatmap(img_array, model, pred_index=None):
    """Generates the Grad-CAM heatmap using the DenseNet base."""
    # 1. Identify the nested base model and its final activation layer
    base_model = model.get_layer('densenet121')
    last_conv_layer_name = 'relu' 
    
    # 2. Create a sub-model that outputs both the feature map and the base model's final output
    inner_model = tf.keras.models.Model(
        [base_model.inputs],
        [base_model.get_layer(last_conv_layer_name).output, base_model.output]
    )

    # 3. Preprocess the image manually (Bypasses the random flip augmentation!)
    preprocessed_img = tf.keras.applications.densenet.preprocess_input(np.copy(img_array))

    # 4. Use GradientTape to record the math operations
    with tf.GradientTape() as tape:
        last_conv_layer_output, base_model_output = inner_model(preprocessed_img)
        tape.watch(last_conv_layer_output)

        preds = base_model_output
        idx = model.layers.index(base_model)
        for layer in model.layers[idx+1:]:
            preds = layer(preds, training=False) 

        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # 5. Compute the gradients
    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # 6. Generate the heatmap
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # 7. Normalize the heatmap
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    
    return heatmap.numpy(), preds[0].numpy()


def generate_gradcam_overlay(original_image, heatmap, alpha=0.6):
    """Overlays the heatmap onto the original image for web display."""
    heatmap_rescaled = np.uint8(255 * heatmap)
    
    jet = plt.colormaps.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap_rescaled]

    jet_heatmap = tf.keras.utils.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((original_image.shape[1], original_image.shape[0]))
    jet_heatmap = tf.keras.utils.img_to_array(jet_heatmap)

    superimposed_img = jet_heatmap * alpha + original_image
    superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)

    superimposed_img_rgb = cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB)
    
    return superimposed_img_rgb