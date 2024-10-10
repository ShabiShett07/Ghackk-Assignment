from textblob import TextBlob
import pandas as pd

# Sample comments (you can replace these with actual comments)
comments = [
    "One of the best articles ever written on the topic. It clearly reflects the differences without any unnecessary details and is really to the point. Great job, Shweta!"
]

# Sentiment analysis using TextBlob
def analyze_sentiment(comment):
    blob = TextBlob(comment)
    polarity = blob.sentiment.polarity
    if polarity > 0:
        return 'Positive'
    elif polarity < 0:
        return 'Negative'
    else:
        return 'Neutral'

# Create a DataFrame for easier analysis
df = pd.DataFrame(comments, columns=['Comment'])
df['Sentiment'] = df['Comment'].apply(analyze_sentiment)

# Filter out neutral comments, keeping only positive and negative
df_filtered = df[df['Sentiment'] != 'Neutral']

# Count the number of each sentiment
sentiment_counts = df_filtered['Sentiment'].value_counts()

# Calculate percentages
total_comments = len(df_filtered)
positive_percent = (sentiment_counts.get('Positive', 0) / total_comments) * 100
negative_percent = (sentiment_counts.get('Negative', 0) / total_comments) * 100

# Print filtered comments and results
print("Filtered Comments with Sentiment:")
print(df_filtered[['Comment', 'Sentiment']])

print(f"\nPositive Comments: {positive_percent:.2f}%")
print(f"Negative Comments: {negative_percent:.2f}%")
