import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt

# ==========================
# Load Dataset
# ==========================
df = pd.read_csv("reviews.csv")

# ==========================
# Sentiment Classification
# ==========================
def analyze_sentiment(review):
    analysis = TextBlob(str(review))
    polarity = analysis.sentiment.polarity

    if polarity > 0:
        return "Positive"
    elif polarity < 0:
        return "Negative"
    else:
        return "Neutral"

# Apply sentiment analysis
df["Sentiment"] = df["Review"].apply(analyze_sentiment)

# ==========================
# Display Results
# ==========================
print("\n===== SENTIMENT ANALYSIS RESULTS =====\n")
print(df)

# Sentiment Summary
sentiment_counts = df["Sentiment"].value_counts()

print("\n===== SUMMARY =====")
print(sentiment_counts)

# Calculate percentages
total_reviews = len(df)

print("\n===== PERCENTAGE DISTRIBUTION =====")
for sentiment, count in sentiment_counts.items():
    percentage = (count / total_reviews) * 100
    print(f"{sentiment}: {percentage:.2f}%")

# ==========================
# Visualization
# ==========================
# ==========================
# Professional Visualization
# ==========================

plt.figure(figsize=(8, 6))

colors = ['#2ECC71', '#E74C3C', '#F1C40F']  # Green, Red, Yellow

sentiment_counts.plot(
    kind='bar',
    color=colors,
    edgecolor='black',
    width=0.6
)

plt.title(
    "Customer Review Sentiment Analysis",
    fontsize=16,
    fontweight='bold'
)

plt.xlabel("Sentiment Category", fontsize=12)
plt.ylabel("Number of Reviews", fontsize=12)

plt.xticks(rotation=0)

# Add values on top of bars
for i, value in enumerate(sentiment_counts):
    plt.text(
        i,
        value + 0.2,
        str(value),
        ha='center',
        fontsize=11,
        fontweight='bold'
    )

plt.grid(axis='y', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()
plt.figure(figsize=(7,7))

plt.pie(
    sentiment_counts,
    labels=sentiment_counts.index,
    autopct='%1.1f%%',
    startangle=90,
    colors=['#2ECC71', '#E74C3C', '#F1C40F'],
    explode=(0.05, 0.05, 0.05)
)

plt.title(
    "Sentiment Distribution",
    fontsize=16,
    fontweight='bold'
)

plt.show()
# ==========================
# Save Results
# ==========================
df.to_csv("sentiment_analysis_results.csv", index=False)

print("\nResults saved to 'sentiment_analysis_results.csv'")