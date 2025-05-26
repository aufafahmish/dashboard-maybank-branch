import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
import ast

# Set page config
st.set_page_config(
    page_title="Maybank Dashboard Analisis Sentimen",
    page_icon="assets/maybank logo round.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

# # Custom CSS
# st.markdown("""
# <style>
#     .main-header {
#         background: linear-gradient(90deg, #1f4e79 0%, #2e6da4 100%);
#         padding: 1rem;
#         border-radius: 10px;
#         color: white;
#         text-align: center;
#         margin-bottom: 2rem;
#     }
#     .metric-card {
#         background: white;
#         padding: 1rem;
#         border-radius: 10px;
#         box-shadow: 0 2px 4px rgba(0,0,0,0.1);
#         border-left: 4px solid #2e6da4;
#     }
#     .stExpander {
#         border: 1px solid #e0e0e0;
#         border-radius: 5px;
#     }
# </style>
# """, unsafe_allow_html=True)
# Custom CSS - Gaya Visual Maybank
# Custom CSS - Tema Cerah ala Maybank
st.markdown("""
<style>
/* ========= GLOBAL BACKGROUND ========= */
body, .main, .block-container {
    background-color: #fffdee !important;
    font-family: 'Arial', sans-serif;
    color: #000000 !important;
}

/* ========= FIX teks default Streamlit ========= */
h1, h2, h3, h4, h5, h6, p, span, label, div[data-testid="stMarkdownContainer"] {
    color: #000000 !important;
}

/* ========= SIDEBAR ========= */
section[data-testid="stSidebar"] {
    background-color: #fffbd1 !important;
}

/* Sidebar dropdown (selectbox): teks dropdown dan opsinya */
.css-1d391kg, .css-1v3fvcr, .css-1cpxqw2 {
    color: #000000 !important;
}
.css-1d391kg > div[role="button"]:hover {
    background-color: #FFE066 !important;
}

/* ========= METRIC COMPONENT ========= */
[data-testid="stMetricValue"] {
    color: #000000 !important;
    font-weight: bold;
}
[data-testid="stMetricLabel"] {
    color: #444444 !important;
    font-weight: bold;
}
[data-testid="stMetricDeltaPositive"] {
    color: #2ecc71 !important; /* hijau naik */
}
[data-testid="stMetricDeltaNegative"] {
    color: #e74c3c !important; /* merah turun */
}
[data-testid="stMetricDeltaNeutral"] {
    color: #888888 !important; /* abu stagnan */
}

/* ========= HEADER UTAMA ========= */
.main-header {
    background: linear-gradient(90deg, #FFD700 0%, #FFB300 100%);
    padding: 1rem;
    border-radius: 10px;
    color: #000000;
    text-align: center;
    font-weight: bold;
    font-size: 1.5rem;
    margin-bottom: 2rem;
}

/* ========= TABS ========= */
div[data-testid="stTabs"] > div > button {
    color: #000000 !important;
    background-color: #fff9d6 !important;
    font-weight: 600;
    border: none;
}
div[data-testid="stTabs"] > div > button[aria-selected="true"] {
    color: #000000 !important;
    border-bottom: 3px solid #FFD700 !important;
    background-color: #fff5b0 !important;
}

/* ========= BUTTON ========= */
button, .stButton > button {
    background-color: #FFD700;
    color: black;
    border-radius: 8px;
    border: none;
    padding: 0.6em 1em;
    font-weight: bold;
}
button:hover {
    background-color: #FFC300;
}

/* ========= EXPANDER ========= */
.stExpander {
    background-color: #FFFBEA;
    border-radius: 5px;
    font-family: 'Arial', sans-serif;
    color: #333;
}

/* ========= INPUT ========= */
input, .stTextInput > div > input {
    border-radius: 5px;
    border: 1px solid #ccc;
    color: black !important;
}

/* ========= DATAFRAME ========= */
.dataframe {
    background-color: white;
    border-radius: 8px;
    border: 1px solid #ddd;
}
</style>
""", unsafe_allow_html=True)




# Load data function with caching
@st.cache_data
def load_data():
    try:
        komen_negatif = pd.read_csv('data/komen negatif.csv')
        komen_netral = pd.read_csv('data/komen netral.csv') 
        komen_positif = pd.read_csv('data/komen positif.csv')
        
        topik_negatif = pd.read_csv('data/topik negatif label.csv')
        topik_netral = pd.read_csv('data/topik netral label.csv')
        topik_positif = pd.read_csv('data/topik positif label.csv')
        
        return {
            'komen_negatif': komen_negatif,
            'komen_netral': komen_netral,
            'komen_positif': komen_positif,
            'topik_negatif': topik_negatif,
            'topik_netral': topik_netral,
            'topik_positif': topik_positif
        }
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Enhanced EDA Functions
def plot_word_count_distribution_st(df, title, color='steelblue'):
    word_counts = df['komentar'].apply(lambda x: len(x.split()) if pd.notna(x) else 0)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(word_counts, bins=20, kde=True, color=color, ax=ax)
    ax.set_title(f'Distribusi Jumlah Kata - {title}', fontsize=16, weight='bold')
    ax.set_xlabel('Jumlah Kata per Komentar', fontsize=12)
    ax.set_ylabel('Frekuensi', fontsize=12)
    plt.tight_layout()
    return fig

def plot_char_count_distribution_st(df, title, color='mediumpurple'):
    char_counts = df['komentar'].apply(lambda x: len(x) if pd.notna(x) else 0)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(char_counts, bins=20, kde=True, color=color, ax=ax)
    ax.set_title(f'Distribusi Panjang Kalimat - {title}', fontsize=16, weight='bold')
    ax.set_xlabel('Jumlah Karakter per Komentar', fontsize=12)
    ax.set_ylabel('Frekuensi', fontsize=12)
    plt.tight_layout()
    return fig

def generate_wordcloud_st(text_series, title, colormap='viridis'):
    text = ' '.join(text_series.dropna().astype(str))
    if not text.strip():
        return None
        
    wc = WordCloud(
        width=800, 
        height=400, 
        background_color='white', 
        colormap=colormap,
        max_words=100
    ).generate(text)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(wc, interpolation='bilinear')
    ax.axis('off')
    ax.set_title(title, fontsize=16, weight='bold', pad=20)
    plt.tight_layout()
    return fig

def get_top_ngrams(corpus, ngram_range=(2, 2), n=10):
    try:
        # Filter out empty/null values
        corpus_clean = [str(text) for text in corpus if pd.notna(text) and str(text).strip()]
        if not corpus_clean:
            return []
            
        vec = CountVectorizer(ngram_range=ngram_range, max_features=1000).fit(corpus_clean)
        bag_of_words = vec.transform(corpus_clean)
        sum_words = bag_of_words.sum(axis=0)
        
        words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
        words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
        return words_freq[:n]
    except Exception as e:
        st.error(f"Error in n-gram analysis: {e}")
        return []

def plot_top_ngrams_st(corpus, title, ngram_range=(2,2), top_n=10, color='skyblue'):
    top_ngrams = get_top_ngrams(corpus, ngram_range=ngram_range, n=top_n)
    
    if not top_ngrams:
        st.warning(f"Tidak ada data untuk {title}")
        return None
        
    ngrams = [i[0] for i in top_ngrams]
    freqs = [i[1] for i in top_ngrams]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    bars = ax.barh(range(len(ngrams)), freqs, color=color)
    
    # Add value labels
    for i, (bar, freq) in enumerate(zip(bars, freqs)):
        ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2, 
                str(freq), va='center', fontweight='bold')
    
    ax.set_yticks(range(len(ngrams)))
    ax.set_yticklabels(ngrams)
    ax.set_title(title, fontsize=16, weight='bold')
    ax.set_xlabel('Frekuensi', fontsize=12)
    ax.invert_yaxis()
    plt.tight_layout()
    return fig

def plot_top_locations_st(df, column='letak', title='Lokasi Terbanyak', top_n=10, color='#5cb85c'):
    if column not in df.columns:
        st.warning(f"Kolom '{column}' tidak ditemukan dalam data")
        return None
        
    lokasi_counts = df[column].value_counts().head(top_n)
    
    if lokasi_counts.empty:
        st.warning("Data lokasi kosong")
        return None
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(range(len(lokasi_counts)), lokasi_counts.values, color=color)
    
    # Add value labels
    for i, (bar, value) in enumerate(zip(bars, lokasi_counts.values)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                str(value), ha='center', va='bottom', fontweight='bold')
    
    ax.set_xticks(range(len(lokasi_counts)))
    ax.set_xticklabels(lokasi_counts.index, rotation=45, ha='right')
    ax.set_title(title, fontsize=16, weight='bold')
    ax.set_xlabel('Lokasi', fontsize=12)
    ax.set_ylabel('Jumlah', fontsize=12)
    plt.tight_layout()
    return fig

def plot_topic_analysis_st(df, title, color="#5cb85c"):
    if df.empty:
        st.warning(f"Data {title} kosong")
        return None
        
    df_sorted = df.sort_values("Count", ascending=True)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    bars = ax.barh(range(len(df_sorted)), df_sorted["Count"], color=color)
    
    # Add value labels
    for i, (bar, value) in enumerate(zip(bars, df_sorted["Count"])):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                str(value), va='center', fontweight='bold', color='black')
    
    ax.set_yticks(range(len(df_sorted)))
    ax.set_yticklabels(df_sorted["Name"], fontsize=10)
    ax.set_title(f"Ringkasan {title}", fontsize=16, weight='bold')
    ax.set_xlabel("Jumlah Komentar", fontsize=12)
    ax.grid(axis='x', linestyle='--', alpha=0.3)
    plt.tight_layout()
    return fig

def show_topic_examples(df, sentiment_type):
    """Show example comments for each topic"""
    if df.empty:
        st.warning("Tidak ada data topik")
        return
        
    for _, row in df.iterrows():
        with st.expander(f"📝 Contoh komentar: {row['Name']} ({row['Count']} komentar)"):
            try:
                # Handle different possible formats for Representative_Docs
                if pd.isna(row['Representative_Docs']):
                    st.write("Tidak ada contoh komentar tersedia")
                    continue
                    
                docs = row['Representative_Docs']
                if isinstance(docs, str):
                    try:
                        docs = ast.literal_eval(docs)
                    except (ValueError, SyntaxError):
                        # If it's already a string, treat as single document
                        docs = [docs]
                elif not isinstance(docs, list):
                    docs = [str(docs)]
                
                if docs and len(docs) > 0:
                    st.write(f"**Jumlah total komentar dalam topik ini: {row['Count']}**")
                    st.write("**Contoh komentar:**")
                    for i, doc in enumerate(docs[:3], 1):  # Show max 3 examples
                        if doc and str(doc).strip():
                            st.write(f"{i}. {doc}")
                else:
                    st.write("Tidak ada contoh komentar tersedia")
                    
            except Exception as e:
                st.error(f"Error menampilkan contoh: {e}")

# Main App
def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>Dashboard of Maybank Branch </h1>
        <p>Dashboard untuk menganalisis sentimen dan topik komentar pelanggan Maybank di seluruh cabang</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load data
    data = load_data()
    if data is None:
        st.error("Gagal memuat data. Pastikan file CSV tersedia di folder 'data/'")
        return
    
    # Sidebar
    st.sidebar.header("Navigasi Dashboard")
    
    # Calculate metrics
    total_negatif = len(data['komen_negatif'])
    total_netral = len(data['komen_netral']) 
    total_positif = len(data['komen_positif'])
    total_komentar = total_negatif + total_netral + total_positif
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Komentar", f"{total_komentar:,}")
    with col2:
        st.metric("Komentar Positif", f"{total_positif:,}", delta=f"{(total_positif/total_komentar*100):.1f}%")
    with col3:
        st.metric("Komentar Netral", f"{total_netral:,}", delta=f"{(total_netral/total_komentar*100):.1f}%")
    with col4:
        st.metric("Komentar Negatif", f"{total_negatif:,}", delta=f"-{(total_negatif/total_komentar*100):.1f}%")
    
    # Main navigation
    analysis_type = st.sidebar.selectbox(
        "Pilih Jenis Analisis:",
        ["Overview", "Sentimen Positif", "Sentimen Netral", "Sentimen Negatif", "Analisis Topik"]
    )
    
    if analysis_type == "Overview":
        st.header("Overview Analisis Sentimen")
        
        # Sentiment distribution pie chart
        col1, col2 = st.columns(2)
        
        with col1:
            fig, ax = plt.subplots(figsize=(8, 8))
            sizes = [total_positif, total_netral, total_negatif]
            labels = ['Positif', 'Netral', 'Negatif']
            colors = ['#2ecc71', '#f39c12', '#e74c3c']
            explode = (0.05, 0.05, 0.05)
            
            wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                                            explode=explode, shadow=True, startangle=90)
            ax.set_title("Distribusi Sentimen Komentar", fontsize=16, weight='bold', pad=20)
            
            # Make percentage text bold and larger
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
                autotext.set_fontsize(12)
            
            st.pyplot(fig)
        
        with col2:
            # Summary statistics
            st.subheader("📈 Statistik Ringkasan")
            
            stats_data = {
                'Sentimen': ['Positif', 'Netral', 'Negatif'],
                'Jumlah': [total_positif, total_netral, total_negatif],
                'Persentase': [f"{(total_positif/total_komentar*100):.1f}%", 
                             f"{(total_netral/total_komentar*100):.1f}%",
                             f"{(total_negatif/total_komentar*100):.1f}%"]
            }
            
            st.dataframe(pd.DataFrame(stats_data), use_container_width=True)
            
            # Key insights
            st.subheader("💡 Insight Utama")
            dominant_sentiment = max([('Positif', total_positif), ('Netral', total_netral), ('Negatif', total_negatif)], 
                                   key=lambda x: x[1])
            
            st.write(f"• Sentimen dominan: **{dominant_sentiment[0]}** ({dominant_sentiment[1]:,} komentar)")
            st.write(f"• Rasio positif vs negatif: **{total_positif/total_negatif:.2f}:1**")
            
            if total_positif > total_negatif:
                st.success("✅ Sentimen pelanggan cenderung positif")
            else:
                st.warning("⚠️ Perlu perhatian pada sentimen negatif")
    
    elif analysis_type == "Sentimen Positif":
        st.header("Analisis Komentar Positif")
        analyze_sentiment_data(data['komen_positif'], "Positif", "#2ecc71")
        
    elif analysis_type == "Sentimen Netral":
        st.header("Analisis Komentar Netral") 
        analyze_sentiment_data(data['komen_netral'], "Netral", "#f39c12")
        
    elif analysis_type == "Sentimen Negatif":
        st.header("Analisis Komentar Negatif")
        analyze_sentiment_data(data['komen_negatif'], "Negatif", "#e74c3c")
        
    elif analysis_type == "Analisis Topik":
        st.header("Analisis Pemodelan Topik")
        
        topic_tab1, topic_tab2, topic_tab3 = st.tabs(["📈 Topik Positif", "📊 Topik Netral", "📉 Topik Negatif"])
        
        with topic_tab1:
            st.subheader("📈 Analisis Topik Positif")
            fig = plot_topic_analysis_st(data['topik_positif'], "Topik Positif", "#2ecc71")
            if fig:
                st.pyplot(fig)
                
                st.subheader("💬 Contoh Komentar per Topik")
                show_topic_examples(data['topik_positif'], "Positif")
        
        with topic_tab2:
            st.subheader("📊 Analisis Topik Netral")
            fig = plot_topic_analysis_st(data['topik_netral'], "Topik Netral", "#f39c12")
            if fig:
                st.pyplot(fig)
                
                st.subheader("💬 Contoh Komentar per Topik")
                show_topic_examples(data['topik_netral'], "Netral")
        
        with topic_tab3:
            st.subheader("📉 Analisis Topik Negatif")
            fig = plot_topic_analysis_st(data['topik_negatif'], "Topik Negatif", "#e74c3c")
            if fig:
                st.pyplot(fig)
                
                st.subheader("💬 Contoh Komentar per Topik")
                show_topic_examples(data['topik_negatif'], "Negatif")

def analyze_sentiment_data(df, sentiment_name, color):
    """Analyze sentiment data with comprehensive visualizations"""
    
    if df.empty:
        st.warning(f"Data komentar {sentiment_name.lower()} kosong")
        return
    
    # Tabs for different analyses
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Distribusi", "☁️ Word Cloud", "📝 Bigram", "📝 Trigram", "📝 Fourgram"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📈 Distribusi Jumlah Kata")
            fig1 = plot_word_count_distribution_st(df, sentiment_name, color)
            if fig1:
                st.pyplot(fig1)
        
        with col2:
            st.subheader("📏 Distribusi Panjang Kalimat")
            fig2 = plot_char_count_distribution_st(df, sentiment_name, color)
            if fig2:
                st.pyplot(fig2)
        
        # Location analysis if column exists
        if 'letak' in df.columns:
            st.subheader("🗺️ Analisis Lokasi")
            fig3 = plot_top_locations_st(df, 'letak', f'Lokasi Terbanyak - {sentiment_name}', color=color)
            if fig3:
                st.pyplot(fig3)
    
    with tab2:
        st.subheader("☁️ Word Cloud")
        fig_wc = generate_wordcloud_st(df['komentar_clean'], f'Word Cloud - {sentiment_name}')
        if fig_wc:
            st.pyplot(fig_wc)
        else:
            st.warning("Tidak dapat membuat word cloud - data kosong atau tidak valid")
    
    with tab3:
        st.subheader("📝 Analisis Bigram (2-kata)")
        fig_bi = plot_top_ngrams_st(df['komentar_clean'], f'Top Bigram - {sentiment_name}', 
                                  ngram_range=(2,2), top_n=15, color=color)
        if fig_bi:
            st.pyplot(fig_bi)
    
    with tab4:
        st.subheader("📝 Analisis Trigram (3-kata)")
        fig_tri = plot_top_ngrams_st(df['komentar_clean'], f'Top Trigram - {sentiment_name}', 
                                   ngram_range=(3,3), top_n=15, color=color)
        if fig_tri:
            st.pyplot(fig_tri)
    
    with tab5:
        st.subheader("📝 Analisis Fourgram (4-kata)")
        fig_four = plot_top_ngrams_st(df['komentar_clean'], f'Top Fourgram - {sentiment_name}', 
                                    ngram_range=(4,4), top_n=15, color=color)
        if fig_four:
            st.pyplot(fig_four)

if __name__ == "__main__":
    main()