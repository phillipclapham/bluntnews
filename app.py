import streamlit as st
import requests
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
from dateutil import parser
from textblob import TextBlob
from collections import Counter
import re
from bs4 import BeautifulSoup
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import opinion_lexicon

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/opinion_lexicon')
except LookupError:
    nltk.download('punkt')
    nltk.download('opinion_lexicon')

# Load environment variables
load_dotenv()
API_KEY = os.getenv('NEWSAPI_KEY')

# Initialize session state for infinite scroll and parameter tracking
if 'articles_displayed' not in st.session_state:
    st.session_state.articles_displayed = 5

if 'prev_params' not in st.session_state:
    st.session_state.prev_params = {}

if 'sources' not in st.session_state:
    st.session_state.sources = []

if 'article_cache' not in st.session_state:
    st.session_state.article_cache = {}

def analyze_bias(text):
    """
    Analyze text for potential bias using multiple indicators
    Returns a dictionary containing bias analysis results
    """
    if not text:
        return {
            "bias_detected": False,
            "bias_score": 0,
            "indicators": [],
            "details": {}
        }

    # Initialize bias indicators
    bias_indicators = []
    details = {}

    # Tokenize text
    sentences = sent_tokenize(text)
    words = word_tokenize(text.lower())

    # Load opinion lexicon
    positive_words = set(opinion_lexicon.positive())
    negative_words = set(opinion_lexicon.negative())

    # 1. Analyze loaded language
    emotional_words = [word for word in words if word in positive_words or word in negative_words]
    emotional_word_ratio = len(emotional_words) / len(words) if words else 0
    if emotional_word_ratio > 0.1:  # More than 10% emotional words
        bias_indicators.append("High emotional language")
        details["emotional_language"] = {
            "ratio": emotional_word_ratio,
            "examples": emotional_words[:5]  # Show first 5 examples
        }

    # 2. Check for qualifying words (indicates potential bias)
    qualifying_words = ['many', 'some', 'few', 'several', 'numerous', 'rarely', 'often', 'generally', 'usually']
    qualifying_count = sum(1 for word in words if word in qualifying_words)
    if qualifying_count > len(sentences) * 0.2:  # More than 20% of sentences contain qualifying words
        bias_indicators.append("Frequent use of qualifying language")
        details["qualifying_language"] = {
            "count": qualifying_count,
            "examples": [word for word in words if word in qualifying_words][:5]
        }

    # 3. Check for source attribution
    source_phrases = ['according to', 'said', 'reported', 'stated', 'claims', 'suggests']
    source_count = sum(1 for phrase in source_phrases if phrase in text.lower())
    if source_count < len(sentences) * 0.1:  # Less than 10% of sentences have source attribution
        bias_indicators.append("Limited source attribution")
        details["source_attribution"] = {
            "count": source_count,
            "total_sentences": len(sentences)
        }

    # 4. Check for absolute language
    absolute_words = ['always', 'never', 'all', 'none', 'every', 'only', 'impossible', 'absolutely']
    absolute_count = sum(1 for word in words if word in absolute_words)
    if absolute_count > 0:
        bias_indicators.append("Use of absolute language")
        details["absolute_language"] = {
            "count": absolute_count,
            "examples": [word for word in words if word in absolute_words]
        }

    # 5. Calculate overall bias score (0-1)
    indicators_count = len(bias_indicators)
    bias_score = min(1.0, indicators_count / 4)  # Normalize to 0-1 range

    return {
        "bias_detected": bool(bias_indicators),
        "bias_score": bias_score,
        "indicators": bias_indicators,
        "details": details
    }

def fetch_article_content(url):
    """Fetch and extract the main content of an article"""
    if url in st.session_state.article_cache:
        return st.session_state.article_cache[url]

    try:
        # Add headers to mimic a browser request
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=5)
        response.raise_for_status()

        # Parse the HTML content
        soup = BeautifulSoup(response.text, 'html.parser')

        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()

        # Find the main content (this might need adjustment based on the website structure)
        # Try common article content selectors
        content = ""
        selectors = [
            "article",
            '[role="article"]',
            ".article-content",
            ".story-content",
            ".post-content",
            "main",
            "#main-content"
        ]

        for selector in selectors:
            if content_element := soup.select_one(selector):
                content = content_element.get_text(strip=True)
                break

        if not content:
            # Fallback: get all paragraph text
            paragraphs = soup.find_all('p')
            content = ' '.join(p.get_text(strip=True) for p in paragraphs)

        # Cache the result
        st.session_state.article_cache[url] = content
        return content

    except Exception as e:
        print(f"Error fetching article content from {url}: {str(e)}")
        return None

def extract_potential_subjects(text):
    """Extract potential subjects using capitalized words and phrases"""
    if not text:
        return []

    # Find sequences of capitalized words
    matches = re.finditer(r'(?:[A-Z][a-zA-Z.]* )*[A-Z][a-zA-Z.]*', text)
    subjects = [match.group() for match in matches if len(match.group()) > 1]

    # Filter out common non-subject capitalized words
    common_words = {'The', 'A', 'An', 'This', 'That', 'These', 'Those', 'I', 'We', 'You', 'He', 'She', 'It', 'They'}
    subjects = [s for s in subjects if s not in common_words]

    return subjects

def identify_main_subject(text, full_content=None):
    """Identify the main subject of the text using capitalization patterns"""
    if not text:
        return None

    # Combine preview text with full content if available
    if full_content:
        text = f"{text} {full_content}"

    # Extract potential subjects
    subjects = extract_potential_subjects(text)

    if not subjects:
        return None

    # Count occurrences and get the most frequent subject
    subject_counts = Counter(subjects)

    # Get the most common subject that's not just a single character
    for subject, _ in subject_counts.most_common():
        if len(subject) > 1:
            return subject

    return None

def simple_sentence_split(text):
    """Simple sentence splitting"""
    # Split on common sentence endings
    sentences = re.split(r'[.!?]+', text)
    return [s.strip() for s in sentences if s.strip()]

def get_targeted_sentiment(text, subject, full_content=None):
    """Analyze sentiment of text specifically towards the identified subject"""
    if not text or not subject:
        return "Neutral", 0

    # Combine preview text with full content if available
    if full_content:
        text = f"{text} {full_content}"

    # Split text into sentences using simple approach
    sentences = simple_sentence_split(text)

    # Find sentences containing the subject
    relevant_sentences = [sent for sent in sentences if subject.lower() in sent.lower()]

    if not relevant_sentences:
        return "Neutral", 0

    # Analyze sentiment of relevant sentences
    sentiments = [TextBlob(sent).sentiment.polarity for sent in relevant_sentences]
    avg_polarity = sum(sentiments) / len(sentiments)

    # Determine sentiment category with more granular thresholds
    if avg_polarity > 0.3:
        return "Very Positive", avg_polarity
    elif 0.1 <= avg_polarity <= 0.3:
        return "Slightly Positive", avg_polarity
    elif -0.1 <= avg_polarity < 0.1:
        return "Neutral", avg_polarity
    elif -0.3 <= avg_polarity < -0.1:
        return "Slightly Negative", avg_polarity
    else:
        return "Very Negative", avg_polarity

def get_sentiment(text):
    """Analyze overall sentiment of text using TextBlob"""
    if not text:
        return "Neutral", 0

    analysis = TextBlob(text)
    polarity = analysis.sentiment.polarity

    if polarity > 0.1:
        return "Positive", polarity
    elif polarity < -0.1:
        return "Negative", polarity
    else:
        return "Neutral", polarity

@st.cache_data(ttl=300)
def fetch_sources(language=None, country=None):
    """Fetch available news sources"""
    params = {
        "apiKey": API_KEY,
        "language": language
    }
    # Only add country if it's provided (to avoid API restrictions)
    if country:
        params["country"] = country

    try:
        response = requests.get("https://newsapi.org/v2/top-headlines/sources", params=params)
        response.raise_for_status()
        data = response.json()
        return data.get('sources', [])
    except requests.exceptions.RequestException as e:
        st.error("An error occurred while fetching sources.")
        print(f"NewsAPI Sources Error: {str(e)}")
        return []

# Cache the API request
@st.cache_data(ttl=300)  # Cache for 5 minutes
def fetch_news(category=None, country=None, language=None, keyword=None, start_date=None, end_date=None, sources=None):
    params = {
        "apiKey": API_KEY,
        "pageSize": 100,
        "language": language,
    }

    # Debug logging
    print(f"Fetching news with params: category={category}, country={country}, sources={sources}")

    # Use everything endpoint for keyword searches or when sources are specified
    if keyword or sources:
        url = "https://newsapi.org/v2/everything"
        # Set a default query if none provided
        params["q"] = keyword if keyword else "news"
        # Add date parameters for the everything endpoint
        params["from"] = start_date.isoformat()
        params["to"] = end_date.isoformat()
        if sources:
            # Convert sources list to comma-separated string
            sources_str = ",".join(str(s) for s in sources)
            params["sources"] = sources_str
            print(f"Using sources: {sources_str}")
    else:
        # Use top-headlines for category/country based searches
        url = "https://newsapi.org/v2/top-headlines"
        if category and category.lower() != 'all':
            params["category"] = category.lower()
        if country:
            params["country"] = country.lower()

    try:
        print(f"Making request to {url} with params: {params}")
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        print(f"API Response status: {response.status_code}")
        if data.get('status') != 'ok':
            print(f"API Error: {data.get('message', 'Unknown error')}")
            st.error(f"API Error: {data.get('message', 'Unknown error')}")
        return data
    except requests.exceptions.RequestException as e:
        error_msg = str(e)
        st.error(f"An error occurred while fetching news data: {error_msg}")
        print(f"NewsAPI Error: {error_msg}")
        if hasattr(e.response, 'text'):
            print(f"Response text: {e.response.text}")
        return None

def display_article(article):
    col1, col2 = st.columns([1, 2])

    with col1:
        if article.get('urlToImage'):
            st.image(article['urlToImage'], use_container_width=True)

    with col2:
        st.subheader(article['title'])
        st.write(f"**Source:** {article['source']['name']}")
        # Safe date parsing
        published_at = "Unknown"
        if article.get('publishedAt'):
            try:
                published_at = parser.parse(article['publishedAt']).strftime('%Y-%m-%d')
            except (ValueError, TypeError):
                pass
        st.write(f"**Published:** {published_at}")

        # Try to fetch full article content
        full_content = None
        with st.spinner('Analyzing article...'):
            try:
                full_content = fetch_article_content(article['url'])
            except Exception as e:
                print(f"Error analyzing full article: {str(e)}")

        # Combine title and description for initial analysis
        preview_text = f"{article.get('title', '')} {article.get('description', '')}"
        analysis_text = full_content if full_content else preview_text

        # Bias Analysis
        bias_results = analyze_bias(analysis_text)
        if bias_results["bias_detected"]:
            st.markdown("### Bias Analysis")
            st.markdown(f"**Bias Score:** {bias_results['bias_score']:.2f}")

            # Display bias indicators with explanations
            st.markdown("**Detected Bias Indicators:**")
            for indicator in bias_results["indicators"]:
                st.markdown(f"- {indicator}")

            # Expandable detailed analysis
            with st.expander("See detailed bias analysis"):
                details = bias_results["details"]

                if "emotional_language" in details:
                    st.markdown("**Emotional Language:**")
                    st.markdown(f"- Ratio: {details['emotional_language']['ratio']:.2f}")
                    st.markdown(f"- Examples: {', '.join(details['emotional_language']['examples'])}")

                if "qualifying_language" in details:
                    st.markdown("**Qualifying Language:**")
                    st.markdown(f"- Count: {details['qualifying_language']['count']}")
                    st.markdown(f"- Examples: {', '.join(details['qualifying_language']['examples'])}")

                if "source_attribution" in details:
                    st.markdown("**Source Attribution:**")
                    st.markdown(f"- Attribution count: {details['source_attribution']['count']}")
                    st.markdown(f"- Total sentences: {details['source_attribution']['total_sentences']}")

                if "absolute_language" in details:
                    st.markdown("**Absolute Language:**")
                    st.markdown(f"- Count: {details['absolute_language']['count']}")
                    st.markdown(f"- Examples: {', '.join(details['absolute_language']['examples'])}")

        # Subject and Sentiment Analysis
        main_subject = identify_main_subject(preview_text, full_content)
        if main_subject:
            st.write(f"**Main Subject:** {main_subject}")
            sentiment, polarity = get_targeted_sentiment(preview_text, main_subject, full_content)
            st.write(f"**Sentiment towards {main_subject}:** {sentiment} (Score: {polarity:.2f})")

            # Indicate if analysis includes full article content
            if full_content:
                st.write("*Analysis based on full article content*")
            else:
                st.write("*Analysis based on article preview*")

            # Visual sentiment indicators with more granular representation
            if sentiment == "Very Positive":
                st.markdown("🟢🟢 Very positive tone")
            elif sentiment == "Slightly Positive":
                st.markdown("🟢 Slightly positive tone")
            elif sentiment == "Neutral":
                st.markdown("⚪ Neutral tone")
            elif sentiment == "Slightly Negative":
                st.markdown("🔴 Slightly negative tone")
            elif sentiment == "Very Negative":
                st.markdown("🔴🔴 Very negative tone")
        else:
            # Fallback to overall sentiment if no main subject is identified
            if article.get('description'):
                sentiment, polarity = get_sentiment(preview_text)
                st.write(article['description'])
                st.write(f"**Overall Sentiment:** {sentiment} (Score: {polarity:.2f})")

                if sentiment == "Positive":
                    st.markdown("🟢 Positive tone")
                elif sentiment == "Negative":
                    st.markdown("🔴 Negative tone")
                else:
                    st.markdown("⚪ Neutral tone")

        st.markdown(f"[Read full article]({article['url']})")

    st.divider()

# Page configuration
st.set_page_config(
    page_title="News App",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar
st.sidebar.title("Customize Your News")

# Source filtering
st.sidebar.markdown("---")
st.sidebar.subheader("Source Filtering")

# Expanded language selection
languages = {
    "English": "en",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Italian": "it",
    "Portuguese": "pt",
    "Russian": "ru",
    "Arabic": "ar",
    "Dutch": "nl",
    "Chinese": "zh"
}
language = st.sidebar.selectbox("Select Language", list(languages.keys()))

# Create tabs for filtering options
filter_type = st.sidebar.radio("Filter By:", ["Category & Country", "Sources"])

if filter_type == "Category & Country":
    # Category selection
    category = st.sidebar.selectbox(
        "Select Category",
        ["All", "Business", "Entertainment", "General", "Health", "Science", "Sports", "Technology"]
    )

    # Country selection
    countries = {
        "United States": "us",
        "United Kingdom": "gb",
        "Canada": "ca",
        "Australia": "au",
        "India": "in",
        "Germany": "de",
        "France": "fr",
        "Italy": "it",
        "Japan": "jp",
        "Brazil": "br",
        "Mexico": "mx",
        "South Africa": "za",
        "Russia": "ru",
        "China": "cn"
    }
    country = st.sidebar.selectbox("Select Country", list(countries.keys()))
    selected_source_ids = None
else:
    # Source selection
    available_sources = fetch_sources(language=languages[language])

    # Create source mapping
    source_mapping = {source['name']: source['id'] for source in available_sources}

    if available_sources:
        selected_sources = st.sidebar.multiselect(
            "Select News Sources",
            options=list(source_mapping.keys()),
            format_func=lambda x: x
        )

        # Map selected source names to their IDs
        selected_source_ids = [source_mapping[name] for name in selected_sources] if selected_sources else None

        if not selected_sources:
            st.sidebar.warning("Please select at least one source")
    else:
        selected_source_ids = None
        st.sidebar.warning("No sources available for selected language")

    # Set these to None when using sources
    category = None
    country = None

# Date selection
st.sidebar.markdown("---")
start_date = st.sidebar.date_input(
    "Start Date",
    value=datetime.today() - timedelta(days=7),
    max_value=datetime.today()
)
end_date = st.sidebar.date_input(
    "End Date",
    value=datetime.today(),
    max_value=datetime.today()
)

# Convert dates to datetime objects for API
start_date = datetime.combine(start_date, datetime.min.time())
end_date = datetime.combine(end_date, datetime.max.time())

# Keyword search
keyword = st.sidebar.text_input("Search Keywords")

# Theme toggle (moved to bottom)
st.sidebar.markdown("---")
theme = st.sidebar.radio("Theme", ["Light", "Dark"])
if theme == "Dark":
    st.markdown("""
        <style>
        .stApp {
            background-color: #1E1E1E;
            color: #FFFFFF;
        }
        </style>
    """, unsafe_allow_html=True)

# Attribution
st.sidebar.markdown("---")
st.sidebar.markdown("Powered by [NewsAPI.org](https://newsapi.org)")

# Main content area
st.title("📰 News Application")
st.write("Stay informed with the latest news based on your preferences.")

# Track current parameters
current_params = {
    'category': category,
    'country': country,
    'language': language,
    'keyword': keyword,
    'start_date': start_date,
    'end_date': end_date,
    'sources': selected_source_ids
}

# Reset articles_displayed when parameters change
if st.session_state.prev_params != current_params:
    st.session_state.articles_displayed = 5
    st.session_state.prev_params = current_params.copy()

# Fetch news with loading indicator
with st.spinner('Fetching news articles...'):
    # Only fetch if we have valid filtering criteria
    should_fetch = (
        (filter_type == "Category & Country") or
        (filter_type == "Sources" and selected_source_ids)
    )

    if should_fetch:
        news_data = fetch_news(
            category=category,
            country=countries.get(country) if country else None,
            language=languages[language],
            keyword=keyword,
            start_date=start_date,
            end_date=end_date,
            sources=selected_source_ids
        )
    else:
        news_data = None

if news_data and news_data.get('articles'):
    articles = news_data['articles']

    # Display articles up to the current count
    for article in articles[:st.session_state.articles_displayed]:
        display_article(article)

    # Show "Load More" button if there are more articles
    if st.session_state.articles_displayed < len(articles):
        if st.button("Load More"):
            st.session_state.articles_displayed += 5
            st.rerun()

    # Show total articles info
    st.write(f"Showing {min(st.session_state.articles_displayed, len(articles))} of {len(articles)} articles")
else:
    st.info("No articles found. Try adjusting your search criteria.")
