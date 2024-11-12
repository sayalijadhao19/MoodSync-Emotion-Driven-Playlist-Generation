from transformers import pipeline

# Load the emotion detection model
emotion_detector = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=True)

# Example text input
text = "I'm feeling fantastic today!"

# Get emotion scores
results = emotion_detector(text)

# Display all emotion scores
for result in results[0]:
    print(f"Emotion: {result['label']}, Score: {result['score']:.2f}")

# Find the primary emotion with the highest score
primary_emotion = max(results[0], key=lambda x: x['score'])
print(f"Primary Emotion: {primary_emotion['label']} with confidence {primary_emotion['score']:.2f}")
