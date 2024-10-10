from textblob import TextBlob
import pandas as pd

comments = [
    "One of the best articles ever written on the topic. It clearly reflects the differences without any unnecessary details and is really to the point. Great job, Shweta!"
]

def analyze_sentiment(comment):
    blob = TextBlob(comment)
    polarity = blob.sentiment.polarity
    if polarity > 0:
        return 'Positive'
    elif polarity < 0:
        return 'Negative'
    else:
        return 'Neutral'

df = pd.DataFrame(comments, columns=['Comment'])
df['Sentiment'] = df['Comment'].apply(analyze_sentiment)

df_filtered = df[df['Sentiment'] != 'Neutral']

sentiment_counts = df_filtered['Sentiment'].value_counts()

total_comments = len(df_filtered)
positive_percent = (sentiment_counts.get('Positive', 0) / total_comments) * 100
negative_percent = (sentiment_counts.get('Negative', 0) / total_comments) * 100

print("Filtered Comments with Sentiment:")
print(df_filtered[['Comment', 'Sentiment']])

print(f"\nPositive Comments: {positive_percent:.2f}%")
print(f"Negative Comments: {negative_percent:.2f}%")
