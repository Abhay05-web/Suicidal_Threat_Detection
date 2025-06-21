import re
import tkinter as tk
from collections import defaultdict
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
import networkx as nx
from PIL import Image, ImageTk
from langdetect import detect, LangDetectException
from typing import Dict, List, Tuple, Optional

# === Trie and NLP Classes ===
class TrieNode:
    def __init__(self):
        self.children: Dict[str, TrieNode] = {}
        self.end_of_phrase: bool = False

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, phrase: str) -> None:
        """Insert a phrase into the trie."""
        node = self.root
        for word in phrase.lower().split():
            if word not in node.children:
                node.children[word] = TrieNode()
            node = node.children[word]
        node.end_of_phrase = True

    def search(self, message: str) -> bool:
        """Search for phrases in the message."""
        words = message.lower().split()
        for i in range(len(words)):
            node = self.root
            j = i
            while j < len(words) and words[j] in node.children:
                node = node.children[words[j]]
                if node.end_of_phrase:
                    return True
                j += 1
        return False

# === Phrase and Sentiment Setup ===
suicidal_phrases: Dict[str, List[str]] = {
    'en': [
        "i want to die", "end my life", "kill myself",
        "i can't go on", "no reason to live", 
        "better off dead", "suicidal thoughts"
    ],
    'es': ["quiero morir", "suicidarme", "acabar con mi vida"]
}

tries: Dict[str, Trie] = {lang: Trie() for lang in suicidal_phrases}
for lang, phrases in suicidal_phrases.items():
    for phrase in phrases:
        tries[lang].insert(phrase)

sentiment_lexicon: Dict[str, Dict[str, int]] = {
    'en': {
        "hopeless": -2, "worthless": -2, "happy": 1, 
        "sad": -1, "die": -3, "death": -3, "suicide": -4,
        "lonely": -2, "depressed": -2, "empty": -1,
        "help": 1, "support": 1, "hope": 2
    },
    'es': {
        "desesperado": -2, "inútil": -2, "feliz": 1,
        "triste": -1, "morir": -3, "muerte": -3, "suicidio": -4
    }
}

language_names: Dict[str, str] = {
    'en': "English",
    'es': "Spanish"
}

# === Analysis Functions ===
def detect_language(text: str) -> str:
    """Detect the language of the text with fallback to English."""
    try:
        lang = detect(text)
        return lang if lang in sentiment_lexicon else 'en'
    except LangDetectException:
        return 'en'

def preprocess_text(text: str) -> str:
    """Clean and normalize text for analysis."""
    text = re.sub(r"[^\w\s']", "", text.lower())  # Keep apostrophes for contractions
    text = re.sub(r"\s+", " ", text).strip()
    return text

def greedy_sentiment_score(text: str, lang: str) -> int:
    """Calculate basic sentiment score."""
    words = preprocess_text(text).split()
    lexicon = sentiment_lexicon.get(lang, {})
    return sum(lexicon.get(word, 0) for word in words)

def enhanced_negation_handler(text: str, lang: str) -> int:
    """Handle negations with context awareness."""
    words = preprocess_text(text).split()
    negation_words = {
        'en': {"not", "no", "never", "nobody", "nothing", "nowhere", "none"},
        'es': {"no", "nunca", "nadie", "nada"}
    }.get(lang, {"not"})
    
    score = 0
    negation_window = 3  # Number of words after negation to consider
    negate_positions = []
    
    # First pass: find all negation positions
    for i, word in enumerate(words):
        if word in negation_words:
            negate_positions.extend(range(i+1, min(i+1+negation_window, len(words))))
    
    # Second pass: calculate score with negation handling
    for i, word in enumerate(words):
        if word in sentiment_lexicon.get(lang, {}):
            if i in negate_positions:
                score -= sentiment_lexicon[lang][word]
            else:
                score += sentiment_lexicon[lang][word]
    
    return score

def build_word_graph(text: str) -> nx.Graph:
    """Build a graph of word connections."""
    words = preprocess_text(text).split()
    G = nx.Graph()
    for i in range(len(words) - 1):
        G.add_edge(words[i], words[i+1])
    return G

def graph_risk_score(G: nx.Graph, lang: str) -> int:
    """Calculate risk score based on word connections."""
    risk_words = {
        'en': {"die", "death", "suicide", "hopeless", "worthless"},
        'es': {"morir", "muerte", "suicidio", "desesperado"}
    }.get(lang, {"die", "suicide"})
    
    # Consider both nodes and edges that connect risk words
    score = sum(1 for node in G.nodes if node in risk_words)
    
    # Additional score for connections between risk words
    for u, v in G.edges:
        if u in risk_words and v in risk_words:
            score += 1
    
    return score

def classify_risk(total_score: int) -> Tuple[str, str]:
    """Classify the risk level with more granular categories."""
    if total_score <= -7:
        return "🔴 Critical Risk", "Immediate intervention required"
    elif -7 < total_score <= -5:
        return "🔴 High Risk", "Urgent attention needed"
    elif -5 < total_score <= -3:
        return "🟠 Moderate-High Risk", "Professional evaluation recommended"
    elif -3 < total_score <= -2:
        return "🟠 Moderate Risk", "Monitor closely"
    elif -2 < total_score < 0:
        return "🟡 Low Risk", "Some concerning elements"
    else:
        return "🟢 No Risk", "No significant risk detected"

def full_analysis_pipeline(text: str, lang: str) -> Tuple[str, int, Dict[str, int]]:
    """Complete analysis pipeline with detailed scoring."""
    # Initialize score components
    scores = {
        'base_sentiment': 0,
        'negation_adjusted': 0,
        'graph_risk': 0,
        'phrase_match': 0
    }
    
    cleaned = preprocess_text(text)
    
    # Check for exact phrase matches first
    if tries[lang].search(cleaned):
        scores['phrase_match'] = -5
    
    # Calculate sentiment scores
    scores['base_sentiment'] = greedy_sentiment_score(text, lang)
    scores['negation_adjusted'] = enhanced_negation_handler(text, lang)
    
    # Graph analysis
    G = build_word_graph(text)
    scores['graph_risk'] = graph_risk_score(G, lang)
    
    # Total score calculation
    total_score = sum(scores.values())
    risk_level, risk_description = classify_risk(total_score)
    
    return risk_level, total_score, scores, risk_description, lang

def analyze_message(message: str) -> Tuple[str, int, Dict[str, int], str, str]:
    """Analyze a message with comprehensive results."""
    cleaned = preprocess_text(message)
    lang = detect_language(cleaned)
    
    if tries[lang].search(cleaned):
        return ("⚠️ Immediate Threat: Suicidal phrase matched", 
                -5, 
                {'phrase_match': -5}, 
                "Exact match with known suicidal phrases detected",
                lang)
    
    risk_level, total_score, scores, risk_description, lang = full_analysis_pipeline(message, lang)
    return risk_level, total_score, scores, risk_description, lang

# === GUI ===
class SuicidalThreatDetectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Suicidal Threat Detection System")
        self.root.geometry("900x700")
        self.style = ttk.Style()
        
        # Configure theme and colors
        self.setup_theme()
        
        # Emergency resources
        self.emergency_resources = {
            'en': {
                'hotline': "National Suicide Prevention Lifeline: 100",
                'website': "https://suicidepreventionlifeline.org",
                'text': "Text HOME to 741741 for crisis support"
            },
            'es': {
                'hotline': "Línea de Prevención del Suicidio: 988",
                'website': "https://suicidepreventionlifeline.org/help-yourself/en-espanol/",
                'text': "Envía 'AYUDA' al 741741 para apoyo en crisis"
            }
        }
        
        # Load logo
        self.logo_img = None
        try:
            img = Image.open("logo.png").resize((120, 120), Image.LANCZOS)
            self.logo_img = ImageTk.PhotoImage(img)
        except FileNotFoundError:
            pass
        
        self.setup_ui()
        self.create_menu()

    def setup_theme(self):
        """Configure the application theme."""
        self.style.theme_use('superhero')  # Changed to superhero theme
        self.style.configure('TFrame', background=self.style.colors.bg)
        self.style.configure('TLabel', font=('Helvetica', 11))
        self.style.configure('TButton', font=('Helvetica', 11, 'bold'))
        self.style.configure('Emergency.TLabel', font=('Helvetica', 11, 'bold'), foreground='red')

    def create_menu(self):
        """Create the application menu bar."""
        menubar = tk.Menu(self.root)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Exit", command=self.root.quit)
        menubar.add_cascade(label="File", menu=file_menu)
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="About", command=self.show_about)
        help_menu.add_command(label="Resources", command=self.show_resources)
        menubar.add_cascade(label="Help", menu=help_menu)
        
        self.root.config(menu=menubar)

    def setup_ui(self):
        """Set up the user interface."""
        self.main_frame = ttk.Frame(self.root, padding=10)
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        # Header with logo and title
        self.header_frame = ttk.Frame(self.main_frame)
        self.header_frame.pack(fill=tk.X, padx=10, pady=10)

        if self.logo_img:
            self.logo_label = ttk.Label(self.header_frame, image=self.logo_img)
            self.logo_label.pack(side=tk.LEFT, padx=10, pady=10)

        self.title_label = ttk.Label(
            self.header_frame, 
            text="Suicidal Threat Detection System",
            font=("Helvetica", 20, "bold")
        )
        self.title_label.pack(side=tk.LEFT, padx=10, pady=20)

        # Content area
        self.content_frame = ttk.Frame(self.main_frame, padding=10)
        self.content_frame.pack(fill=tk.BOTH, expand=True)

        # Input card
        self.input_card = ttk.LabelFrame(
            self.content_frame, 
            text="Input Message", 
            padding=15,
            bootstyle="dark"
        )
        self.input_card.pack(fill=tk.X, pady=(0, 20))

        self.input_label = ttk.Label(
            self.input_card, 
            text="Enter message to analyze:", 
            font=("Helvetica", 12, "bold")
        )
        self.input_label.pack(anchor=tk.W, padx=10, pady=(0, 10))

        self.message_entry = tk.Text(
            self.input_card, 
            height=8, 
            font=("Helvetica", 12), 
            wrap=tk.WORD,
            padx=10,
            pady=10
        )
        self.message_entry.pack(fill=tk.X, padx=10, pady=(0, 10))

        # Button frame
        self.button_frame = ttk.Frame(self.input_card)
        self.button_frame.pack(fill=tk.X, pady=(5, 0))

        self.analyze_btn = ttk.Button(
            self.button_frame, 
            text="Analyze Message", 
            command=self.analyze_message, 
            bootstyle="primary",
            width=15
        )
        self.analyze_btn.pack(side=tk.LEFT, padx=(0, 10))

        self.clear_btn = ttk.Button(
            self.button_frame, 
            text="Clear", 
            command=self.clear_input, 
            bootstyle="secondary",
            width=10
        )
        self.clear_btn.pack(side=tk.LEFT)

        # Results card
        self.results_card = ttk.LabelFrame(
            self.content_frame, 
            text="Analysis Results", 
            padding=15,
            bootstyle="dark"
        )
        self.results_card.pack(fill=tk.BOTH, expand=True)

        # Results notebook for tabs
        self.results_notebook = ttk.Notebook(self.results_card)
        self.results_notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Summary tab
        self.summary_tab = ttk.Frame(self.results_notebook)
        self.results_notebook.add(self.summary_tab, text="Summary")

        self.summary_text = tk.Text(
            self.summary_tab,
            height=8,
            font=("Helvetica", 12),
            wrap=tk.WORD,
            padx=10,
            pady=10,
            state=tk.DISABLED
        )
        self.summary_text.pack(fill=tk.BOTH, expand=True)

        # Detailed scores tab
        self.details_tab = ttk.Frame(self.results_notebook)
        self.results_notebook.add(self.details_tab, text="Details")

        self.details_text = tk.Text(
            self.details_tab,
            height=8,
            font=("Helvetica", 12),
            wrap=tk.WORD,
            padx=10,
            pady=10,
            state=tk.DISABLED
        )
        self.details_text.pack(fill=tk.BOTH, expand=True)

        # Emergency resources frame (hidden by default)
        self.emergency_frame = ttk.Frame(
            self.results_card,
            padding=10,
            bootstyle="danger"
        )
        self.emergency_label = ttk.Label(
            self.emergency_frame,
            text="",
            font=("Helvetica", 12, "bold"),
            wraplength=600,
            justify=tk.LEFT
        )
        self.emergency_label.pack(fill=tk.X)
        self.emergency_frame.pack_forget()  # Hide until needed

        # Status bar
        self.status_bar = ttk.Label(
            self.main_frame, 
            text="Ready", 
            anchor=tk.W, 
            font=("Helvetica", 10),
            bootstyle="inverse-dark"
        )
        self.status_bar.pack(fill=tk.X, padx=10, pady=(0, 10))

    def analyze_message(self):
        """Analyze the entered message and display results."""
        msg = self.message_entry.get("1.0", tk.END).strip()
        if not msg:
            self.update_status("Please enter a message to analyze.")
            return

        self.update_status("Analyzing message...")
        self.root.update_idletasks()  # Force UI update

        try:
            risk_level, total_score, scores, risk_description, lang = analyze_message(msg)
            
            # Update summary tab
            self.summary_text.config(state=tk.NORMAL)
            self.summary_text.delete("1.0", tk.END)
            
            summary_content = (
                f"Risk Level: {risk_level}\n\n"
                f"Description: {risk_description}\n\n"
                f"Total Risk Score: {total_score}\n"
                f"Language Detected: {language_names.get(lang, lang)}\n"
            )
            
            self.summary_text.insert(tk.END, summary_content)
            
            # Color coding based on risk level
            if "Critical" in risk_level or "High" in risk_level:
                self.summary_text.tag_add('danger', '1.0', '1.end')
                self.summary_text.tag_config('danger', foreground='red')
            elif "Moderate" in risk_level:
                self.summary_text.tag_add('warning', '1.0', '1.end')
                self.summary_text.tag_config('warning', foreground='orange')
            
            self.summary_text.config(state=tk.DISABLED)
            
            # Update details tab
            self.details_text.config(state=tk.NORMAL)
            self.details_text.delete("1.0", tk.END)
            
            details_content = (
                "Detailed Score Breakdown:\n\n"
                f"Base Sentiment: {scores.get('base_sentiment', 0)}\n"
                f"Negation Adjusted: {scores.get('negation_adjusted', 0)}\n"
                f"Graph Risk Score: {scores.get('graph_risk', 0)}\n"
                f"Phrase Match: {scores.get('phrase_match', 0)}\n\n"
                "Interpretation:\n"
                "- Base Sentiment: Raw sentiment score from word analysis\n"
                "- Negation Adjusted: Impact of negations on sentiment\n"
                "- Graph Risk: Connections between risky words\n"
                "- Phrase Match: Exact matches with known suicidal phrases"
            )
            
            self.details_text.insert(tk.END, details_content)
            self.details_text.config(state=tk.DISABLED)
            
            # Show/hide emergency resources based on risk
            if "Critical" in risk_level or "High" in risk_level:
                self.show_emergency_resources(lang)
            else:
                self.emergency_frame.pack_forget()
            
            self.update_status("Analysis complete.")
            
        except Exception as e:
            self.update_status(f"Error during analysis: {str(e)}")
            self.summary_text.config(state=tk.NORMAL)
            self.summary_text.delete("1.0", tk.END)
            self.summary_text.insert(tk.END, f"Error: {str(e)}")
            self.summary_text.config(state=tk.DISABLED)

    def show_emergency_resources(self, lang: str = 'en'):
        """Display emergency resources for high-risk cases."""
        resources = self.emergency_resources.get(lang, self.emergency_resources['en'])
        
        resource_text = (
            "❗ IMMEDIATE SUPPORT AVAILABLE ❗\n\n"
            f"Hotline: {resources['hotline']}\n"
            f"Crisis Text: {resources['text']}\n"
            f"Website: {resources['website']}"
        )
        
        self.emergency_label.config(text=resource_text)
        self.emergency_frame.pack(fill=tk.X, pady=(10, 0))

    def clear_input(self):
        """Clear the input and results."""
        self.message_entry.delete("1.0", tk.END)
        self.summary_text.config(state=tk.NORMAL)
        self.summary_text.delete("1.0", tk.END)
        self.summary_text.config(state=tk.DISABLED)
        self.details_text.config(state=tk.NORMAL)
        self.details_text.delete("1.0", tk.END)
        self.details_text.config(state=tk.DISABLED)
        self.emergency_frame.pack_forget()
        self.update_status("Ready")

    def update_status(self, msg: str):
        """Update the status bar message."""
        self.status_bar.config(text=msg)
        self.root.update_idletasks()

    def show_about(self):
        """Show the about dialog."""
        about_text = (
            "Suicidal Threat Detection System\n"
            "Version 2.0\n\n"
            "This tool analyzes text for potential suicidal ideation\n"
            "using natural language processing techniques.\n\n"
            "For emergency situations, please contact local\n"
            "crisis support services immediately."
        )
        
        ttk.dialogs.Messagebox.show_info(
            about_text,
            title="About",
            parent=self.root
        )

    def show_resources(self):
        """Show the resources dialog."""
        resources_text = (
            "Emergency Resources:\n\n"
            "United States:\n"
            "National Suicide Prevention Lifeline: 988\n"
            "Crisis Text Line: Text HOME to 741741\n\n"
            "International:\n"
            "Find local resources at: https://www.iasp.info/resources/Crisis_Centres/"
        )
        
        ttk.dialogs.Messagebox.show_info(
            resources_text,
            title="Crisis Resources",
            parent=self.root
        )

if __name__ == "__main__":
    root = ttk.Window(themename="superhero")  # Changed from "darkly" to "superhero"
    app = SuicidalThreatDetectorApp(root)
    root.mainloop()