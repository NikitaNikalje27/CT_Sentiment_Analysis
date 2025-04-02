import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from wordcloud import WordCloud

# Download VADER for sentiment analysis
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

# Sample E-Commerce Reviews Dataset
data = {
    "Review_ID": [1, 2, 3, 4, 5],
    "Customer_Name": ["Alice", "Bob", "Charlie", "David", "Emma"],
    "Review_Text": [
        "The product is amazing! Works perfectly.",
        "Not great, had some issues with the quality.",
        "Absolutely love it! Best purchase ever.",
        "Terrible experience, never buying again.",
        "It's okay, but could be better."
    ]
}

df = pd.DataFrame(data)

# Function to analyze sentiment
def get_sentiment(text):
    score = sia.polarity_scores(text)
    if score['compound'] >= 0.05:
        return "Positive"
    elif score['compound'] <= -0.05:
        return "Negative"
    else:
        return "Neutral"

# Apply sentiment analysis
df['Sentiment'] = df['Review_Text'].apply(get_sentiment)

# Visualize sentiment distribution
plt.figure(figsize=(6, 4))
sns.countplot(x=df['Sentiment'], palette='coolwarm')
plt.title("Sentiment Analysis of Reviews")
plt.xlabel("Sentiment")
plt.ylabel("Count")
plt.show()

# Generate Word Cloud
text = " ".join(review for review in df.Review_Text)
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

# Display results
df
