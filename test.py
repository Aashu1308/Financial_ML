from tensorflow.keras.models import load_model
import joblib

model = load_model('model_hugging/fuzz_dnn_full_model.keras')
print("Model loaded successfully:", model.summary())
scaler = joblib.load('model_hugging/fuzzy_dnn_scaler.pkl')
print("Scaler loaded successfully:", scaler)

model.save('model_hugging/fuzz_dnn_full_model.keras')
joblib.dump(scaler, 'model_hugging/fuzzy_dnn_scaler.pkl')

print("Files re-saved successfully")

# Save in SavedModel format
save_path = 'model_hugging/fuzz_dnn_full_model'
model.export(save_path)

print("Model saved in SavedModel format")
