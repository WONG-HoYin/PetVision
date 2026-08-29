import tensorflow as tf
import cv2
import numpy as np
import httpx
import asyncio

def load_pet_model(model_path):

    return tf.keras.models.load_model(model_path, compile=False)

def process_image(image, size=224):
    img_array = np.array(image.resize((size, size))) / 255.0
    return np.expand_dims(img_array, axis=0)

def get_gradcam_heatmap(model, img_input, last_conv_layer, pred_index):
    try:

        grad_model = tf.keras.models.Model(
            inputs=[model.inputs],
            outputs=[model.get_layer(last_conv_layer).output, model.output]
        )
        with tf.GradientTape() as tape:
            conv_outputs, preds = grad_model(img_input)
            class_channel = preds[1][:, pred_index]
        
        grads = tape.gradient(class_channel, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        heatmap = conv_outputs[0] @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap).numpy()
        heatmap = np.maximum(heatmap, 0)
        if np.max(heatmap) != 0: heatmap /= np.max(heatmap)
        heatmap[heatmap < 0.2] = 0
        return heatmap
    except Exception as e:
        print(f"Grad-CAM Error: {e}")
        return np.zeros((7, 7))

def overlay_heatmap(original_img, heatmap):
    heatmap_resized = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
    heatmap_255 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_255, cv2.COLORMAP_JET)
    heatmap_color_rgb = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(original_img, 0.7, heatmap_color_rgb, 0.3, 0)
