import pandas as pd
import asyncio
from asyncio import Semaphore
from openai import AsyncAzureOpenAI
import numpy as np
import random
import json
import time
import os
from typing import List, Dict, Any

# Initialize Azure OpenAI client
async_gpt4o_client = AsyncAzureOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
)

SEM_LIMIT = 50  # Rate limiting for generation
semaphore = Semaphore(SEM_LIMIT)

# Configuration parameters
LANGUAGES = ['English', 'Hindi', 'Marathi', 'Tamil', 'Telugu', 'Malayalam', 'Kannada']
EMOTIONS = [
    'empathy', 'politeness', 'apology', 'patience', 'attentiveness',
    'concern', 'cheerfulness', 'neutrality', 'reassurance', 'gratitude',
    'proactiveness', 'enthusiasm', 'persuasiveness', 'confidence', 'urgency'
]

INDUSTRIES = [
    'BFSI', 'Healthcare', 'Manufacturing', 'Quick_Commerce', 'EdTech',
    'Real_Estate', 'Retail_Marketplace', 'Automotive', 'Hospitality'
]

# Speech rate constants
WORDS_PER_MINUTE = 140
WORDS_PER_HOUR = WORDS_PER_MINUTE * 60  # 8400 words per hour
N_HOURS_PER_LANGUAGE = 0.25

# Customer address types for variation
CUSTOMER_TYPES = ['male_customer', 'female_customer', 'neutral_customer']

# NEW PARAMETER: Number of sentences per GPT call
N_SENTENCES_PER_CALL = 25  # Adjust this value as needed

# Updated Language-specific instructions for female-oriented code-switching with transliterated English
FEMALE_CODESWITCHING_INSTRUCTIONS = {
    "English": "Generate pure English sentences with proper grammar and natural flow.",
    
    "Hindi": """Create sentences with Hindi as the PRIMARY language framework using Devanagari script.
BALANCED MIXING REQUIREMENT: Include exactly 1-2 English concepts per sentence, but write them in Devanagari script (transliterated).
This should feel natural like real Indian call center conversations where English concepts are adopted into Hindi script.
TARGET: 85-90% Hindi + 10-15% transliterated English concepts (1-2 words)

GOOD EXAMPLES:
- "आपका इश्यू सॉल्व हो गया है।" (issue→इश्यू, solve→सॉल्व)
- "यह ऑफर आज के लिए है।" (offer→ऑफर)
- "आपकी कम्प्लेंट प्रोसेस करेंगे।" (complaint→कम्प्लेंट, process→प्रोसेस)
- "आपकी समस्या का समाधान हमारी टीम जल्दी से करने की कोशिश कर रही है, कृपया थोड़ा सर्विस का इंतजार करें।" (service→सर्विस)
- "हमारे पास आज बहुत सारे अच्छे उत्पाद उपलब्ध हैं, आप कोई भी प्रोडक्ट चुन सकते हैं।" (product→प्रोडक्ट)
- "आपका खाता सफलतापूर्वक बनाया गया है और अब आप अपनी सभी सेवाओं का उपयोग कर सकते हैं, कोई प्रॉब्लम नहीं है।" (problem→प्रॉब्लम)

TRANSLITERATION GUIDE:
- issue → इश्यू, problem → प्रॉब्लम, service → सर्विस, offer → ऑफर
- product → प्रोडक्ट, support → सपोर्ट, account → अकाउंट, process → प्रोसेस
- complaint → कम्प्लेंट, update → अपडेट, system → सिस्टम, online → ऑनलाइन

AVOID: Roman script English words - all concepts should be in Devanagari script
Write the ENTIRE sentence in Devanagari script, including transliterated English concepts.""",

    "Marathi": """Create sentences with Marathi as the PRIMARY language framework using Devanagari script.
BALANCED MIXING REQUIREMENT: Include exactly 1-2 English concepts per sentence, but write them in Devanagari script (transliterated).
TARGET: 85-90% Marathi + 10-15% transliterated English concepts (1-2 words)

GOOD EXAMPLES:
- "तुमची कम्प्लेंट आम्ही करू।" (complaint→कम्प्लेंट)
- "हा प्रॉडक्ट खूप पॉप्युलर आहे।" (product→प्रॉडक्ट, popular→पॉप्युलर)
- "आज स्पेशल प्राईस आहे।" (special→स्पेशल, price→प्राईस)
- "तुमच्या खात्यातील सर्व माहिती योग्य प्रकारे अपडेट झाली आहे, आता तुम्ही सर्व सर्विस वापरू शकता।" (update→अपडेट, service→सर्विस)
- "आमच्या कंपनीकडे खूप छान आणि दर्जेदार उत्पादन उपलब्ध आहेत, तुम्हाला कोणता प्रॉडक्ट आवडेल?" (product→प्रॉडक्ट)
- "तुमची विनंती आम्हाला मिळाली आहे आणि आमची टीम लवकरच तुमच्याशी संपर्क साधेल, कृपया सपोर्ट साठी धन्यवाद।" (team→टीम, support→सपोर्ट)

TRANSLITERATION GUIDE:
- service → सर्विस, team → टीम, support → सपोर्ट, update → अपडेट
- product → प्रॉडक्ट, account → अकाउंट, system → सिस्टम, online → ऑनलाइन

Write the ENTIRE sentence in Devanagari script, with transliterated English concepts.""",

    "Tamil": """Create sentences with Tamil as the PRIMARY language framework using Tamil script.
BALANCED MIXING REQUIREMENT: Include exactly 1-2 English concepts per sentence, but write them in Tamil script (transliterated).
TARGET: 85-90% Tamil + 10-15% transliterated English concepts (1-2 words)

GOOD EXAMPLES:
- "உங்கள் ப்ராப்ளம் சால்வ் ஆச்சு।" (problem→ப்ராப்ளம், solve→சால்வ்)
- "இந்த ஆஃபர் லிமிடெட் டைம்।" (offer→ஆஃபர், limited→லிமிடெட், time→டைம்)
- "நம்ம சர்வீஸ் நல்லா இருக்கா?" (service→சர்வீஸ்)
- "உங்கள் கணக்கு முழுமையாக தயார் ஆகிவிட்டது, இப்போது நீங்கள் எல்லா வசதிகளையும் பயன்படுத்தலாம், ஏதாவது ப்ராப்ளம் இருந்தா சொல்லுங்க।" (problem→ப்ராப்ளம்)
- "எங்கள் கம்பெனியில் மிகவும் நல்ல தரமான பொருட்கள் கிடைக்கின்றன, நீங்கள் எந்த ப்ராடக்ட் விரும்புவீர்கள்?" (company→கம்பெனி, product→ப்ராடக்ட்)
- "உங்கள் புகார் எங்களுக்கு கிடைத்துவிட்டது, எங்கள் டீம் விரைவில் இதை தீர்த்து வைக்கும், உங்கள் சப்போர்ட் க்கு நன்றி।" (team→டீம், support→சப்போர்ட்)

TRANSLITERATION GUIDE:
- problem → ப்ராப்ளம், service → சர்வீஸ், offer → ஆஃபர், product → ப்ராடக்ட்
- support → சப்போர்ட், account → அக்கவுண்ட், system → சிஸ்டம், team → டீம்

Write the ENTIRE sentence in Tamil script, with transliterated English concepts.""",

    "Telugu": """Create sentences with Telugu as the PRIMARY language framework using Telugu script.
BALANCED MIXING REQUIREMENT: Include exactly 1-2 English concepts per sentence, but write them in Telugu script (transliterated).
TARGET: 85-90% Telugu + 10-15% transliterated English concepts (1-2 words)

GOOD EXAMPLES:
- "మీ ఇష్యూ ఫిక్స్ చేసాము।" (issue→ఇష్యూ, fix→ఫిక్స్)
- "ఈ ప్రోడక్ట్ చాలా డిమాండ్ లో ఉంది।" (product→ప్రోడక్ట్, demand→డిమాండ్)
- "మా సర్వీస్ ఎలా అనిపించింది?" (service→సర్వీస్)
- "మీ ఖాతా పూర్తిగా సిద్ధం అయ్యింది, ఇప్పుడు మీరు అన్ని సౌకర్యాలను ఉపయోగించవచ్చు, ఏదైనా ప్రాబ్లమ్ ఉంటే చెప్పండి।" (problem→ప్రాబ్లమ్)
- "మా కంపెనీలో చాలా మంచి నాణ్యతతో కూడిన వస్తువులు అందుబాటులో ఉన్నాయి, మీరు ఏ ప్రోడక్ట్ కావాలని అనుకుంటున్నారు?" (company→కంపెనీ, product→ప్రోడక్ట్)
- "మీ ఫిర్యాదు మాకు చేరింది, మా టీమ్ త్వరగా దీనిని పరిష్కరిస్తారు, మీ సపోర్ట్ కు ధన్యవాదాలు।" (team→టీమ్, support→సపోర్ట్)

TRANSLITERATION GUIDE:
- problem → ప్రాబ్లమ్, service → సర్వీస్, team → టీమ్, support → సపోర్ట్
- product → ప్రోడక్ట్, account → అకౌంట్, system → సిస్టమ్, online → ఆన్లైన్

Write the ENTIRE sentence in Telugu script, with transliterated English concepts.""",

    "Malayalam": """Create sentences with Malayalam as the PRIMARY language framework using Malayalam script.
BALANCED MIXING REQUIREMENT: Include exactly 1-2 English concepts per sentence, but write them in Malayalam script (transliterated).
TARGET: 85-90% Malayalam + 10-15% transliterated English concepts (1-2 words)

GOOD EXAMPLES:
- "നിങ്ങളുടെ കംപ്ലയിന്റ് പ്രോസസ്സ് ചെയ്യാം।" (complaint→കംപ്ലയിന്റ്, process→പ്രോസസ്സ്)
- "ഈ പ്രൊഡക്ട് നല്ല ക്വാളിറ്റി ആണ്।" (product→പ്രൊഡക്ട്, quality→ക്വാളിറ്റി)
- "എങ്ങനെ ഉണ്ട് ഞങ്ങളുടെ സർവീസ്?" (service→സർവീസ്)
- "നിങ്ങളുടെ അക്കൗണ്ട് പൂർണ്ണമായി തയ്യാറാക്കി കഴിഞ്ഞു, ഇപ്പോൾ നിങ്ങൾക്ക് എല്ലാ സൗകര്യങ്ങളും ഉപയോഗിക്കാം, എന്തെങ്കിലും പ്രോബ്ലം ഉണ്ടെങ്കിൽ പറയൂ।" (account→അക്കൗണ്ട്, problem→പ്രോബ്ലം)
- "ഞങ്ങളുടെ കമ്പനിയിൽ വളരെ നല്ല നിലവാരമുള്ള സാധനങ്ങൾ ലഭ്യമാണ്, നിങ്ങൾക്ക് ഏത് പ്രൊഡക്ട് വേണം?" (company→കമ്പനി, product→പ്രൊഡക്ട്)
- "നിങ്ങളുടെ പരാതി ഞങ്ങൾക്ക് ലഭിച്ചു, ഞങ്ങളുടെ ടീം ഉടൻ ഇത് പരിഹരിക്കും, നിങ്ങളുടെ സപ്പോർട്ട് നു നന്ദി।" (team→ടീം, support→സപ്പോർട്ട്)

TRANSLITERATION GUIDE:
- problem → പ്രോബ്ലം, service → സർവീസ്, team → ടീം, support → സപ്പോർട്ട്
- product → പ്രൊഡക്ട്, account → അക്കൗണ്ട്, system → സിസ്റ്റം, online → ഓൺലൈൻ

Write the ENTIRE sentence in Malayalam script, with transliterated English concepts.""",

    "Kannada": """Create sentences with Kannada as the PRIMARY language framework using Kannada script.
BALANCED MIXING REQUIREMENT: Include exactly 1-2 English concepts per sentence, but write them in Kannada script (transliterated).
TARGET: 85-90% Kannada + 10-15% transliterated English concepts (1-2 words)

GOOD EXAMPLES:
- "ನಿಮ್ಮ ಪ್ರಾಬ್ಲಮ್ ಸಾಲ್ವ್ ಮಾಡ್ತೀವಿ।" (problem→ಪ್ರಾಬ್ಲಮ್, solve→ಸಾಲ್ವ್)
- "ಈ ಆಫರ್ ಲಿಮಿಟೆಡ್ ಟೈಮ್ ಗೆ ಮಾತ್ರ।" (offer→ಆಫರ್, limited→ಲಿಮಿಟೆಡ್, time→ಟೈಮ್)
- "ನಮ್ಮ ಸರ್ವೀಸ್ ಎಷ್ಟು ಚೆನ್ನಾಗಿತ್ತು?" (service→ಸರ್ವೀಸ್)
- "ನಿಮ್ಮ ಖಾತೆ ಸಂಪೂರ್ಣವಾಗಿ ಸಿದ್ಧವಾಗಿದೆ, ಈಗ ನೀವು ಎಲ್ಲಾ ಸೌಕರ್ಯಗಳನ್ನು ಬಳಸಬಹುದು, ಏನಾದರೂ ಪ್ರಾಬ್ಲಮ್ ಇದ್ದರೆ ಹೇಳಿ।" (problem→ಪ್ರಾಬ್ಲಮ್)
- "ನಮ್ಮ ಕಂಪನಿಯಲ್ಲಿ ಬಹಳ ಒಳ್ಳೆಯ ಗುಣಮಟ್ಟದ ವಸ್ತುಗಳು ಲಭ್ಯವಿದೆ, ನೀವು ಯಾವ ಪ್ರೊಡಕ್ಟ್ ಬೇಕು?" (company→ಕಂಪನಿ, product→ಪ್ರೊಡಕ್ಟ್)
- "ನಿಮ್ಮ ದೂರು ನಮಗೆ ಬಂದಿದೆ, ನಮ್ಮ ಟೀಮ್ ಬೇಗನೆ ಇದನ್ನು ಪರಿಹರಿಸುತ್ತದೆ, ನಿಮ್ಮ ಸಪೋರ್ಟ್ ಗೆ ಧನ್ಯವಾದಗಳು।" (team→ಟೀಮ್, support→ಸಪೋರ್ಟ್)

TRANSLITERATION GUIDE:
- problem → ಪ್ರಾಬ್ಲಮ್, service → ಸರ್ವೀಸ್, team → ಟೀಮ್, support → ಸಪೋರ್ಟ್
- product → ಪ್ರೊಡಕ್ಟ್, account → ಅಕೌಂಟ್, system → ಸಿಸ್ಟಮ್, online → ಆನ್ಲೈನ್

Write the ENTIRE sentence in Kannada script, with transliterated English concepts."""
}

def calculate_word_count(sentence: str) -> int:
    """Calculate word count manually using simple split method"""
    if not sentence or not sentence.strip():
        return 0
    return len(sentence.strip().split())

def select_customer_type() -> str:
    """Select customer type for addressee variation"""
    return random.choice(CUSTOMER_TYPES)

def generate_sentence_parameters(n_sentences: int) -> List[Dict]:
    """Generate parameters for multiple sentences"""
    sentence_params = []
    for i in range(n_sentences):
        params = {
            'target_length': int(np.random.normal(25, 15)),  # Gaussian distribution
            'emotion': random.choice(EMOTIONS),
            'industry': random.choice(INDUSTRIES),
            'use_complex': random.random() < 0.1,  # 10% chance
            'customer_type': select_customer_type()
        }
        # Clamp target length between 2 and 70
        params['target_length'] = max(2, min(70, params['target_length']))
        sentence_params.append(params)
    return sentence_params

def create_comprehensive_analysis(sentences: List[Dict[str, Any]]) -> Dict:
    """Create comprehensive distribution analysis with percentiles"""
    df = pd.DataFrame(sentences)
    
    # Define percentiles for detailed analysis
    percentiles = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
    
    analysis = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'total_sentences': len(df),
        'total_words': df['actual_word_count'].sum(),
        'total_characters': df['sentence_char_length'].sum(),
        'estimated_speech_hours': df['actual_word_count'].sum() / WORDS_PER_HOUR
    }
    
    # Overall character length distribution
    char_stats = df['sentence_char_length'].describe(percentiles=percentiles)
    analysis['character_length_distribution'] = {
        'overall': char_stats.to_dict(),
        'by_language': {},
        'by_emotion': {},
        'by_industry': {},
        'by_complex_words': {}
    }
    
    # Overall word count distribution
    word_stats = df['actual_word_count'].describe(percentiles=percentiles)
    analysis['word_count_distribution'] = {
        'overall': word_stats.to_dict(),
        'by_language': {},
        'by_emotion': {},
        'by_industry': {},
        'by_complex_words': {}
    }
    
    # Character length by language
    for lang in df['language'].unique():
        lang_df = df[df['language'] == lang]
        char_stats_lang = lang_df['sentence_char_length'].describe(percentiles=percentiles)
        word_stats_lang = lang_df['actual_word_count'].describe(percentiles=percentiles)
        analysis['character_length_distribution']['by_language'][lang] = char_stats_lang.to_dict()
        analysis['word_count_distribution']['by_language'][lang] = word_stats_lang.to_dict()
    
    # Character length by emotion
    for emotion in df['emotion'].unique():
        emotion_df = df[df['emotion'] == emotion]
        char_stats_emotion = emotion_df['sentence_char_length'].describe(percentiles=percentiles)
        word_stats_emotion = emotion_df['actual_word_count'].describe(percentiles=percentiles)
        analysis['character_length_distribution']['by_emotion'][emotion] = char_stats_emotion.to_dict()
        analysis['word_count_distribution']['by_emotion'][emotion] = word_stats_emotion.to_dict()
    
    # Character length by industry
    for industry in df['industry'].unique():
        industry_df = df[df['industry'] == industry]
        char_stats_industry = industry_df['sentence_char_length'].describe(percentiles=percentiles)
        word_stats_industry = industry_df['actual_word_count'].describe(percentiles=percentiles)
        analysis['character_length_distribution']['by_industry'][industry] = char_stats_industry.to_dict()
        analysis['word_count_distribution']['by_industry'][industry] = word_stats_industry.to_dict()
    
    # Character length by complex words usage
    for complex_flag in [True, False]:
        complex_df = df[df['use_complex_words'] == complex_flag]
        if len(complex_df) > 0:
            char_stats_complex = complex_df['sentence_char_length'].describe(percentiles=percentiles)
            word_stats_complex = complex_df['actual_word_count'].describe(percentiles=percentiles)
            key = 'with_complex_words' if complex_flag else 'without_complex_words'
            analysis['character_length_distribution']['by_complex_words'][key] = char_stats_complex.to_dict()
            analysis['word_count_distribution']['by_complex_words'][key] = word_stats_complex.to_dict()
    
    # Target vs Actual length analysis
    df['length_difference'] = df['actual_word_count'] - df['target_length']
    df['length_accuracy'] = np.abs(df['length_difference']) <= 3  # Within 3 words is considered accurate
    
    analysis['target_vs_actual'] = {
        'mean_difference': df['length_difference'].mean(),
        'std_difference': df['length_difference'].std(),
        'accuracy_rate': df['length_accuracy'].mean(),
        'accuracy_count': df['length_accuracy'].sum(),
        'difference_distribution': df['length_difference'].describe(percentiles=percentiles).to_dict()
    }
    
    # Correlation analysis
    analysis['correlations'] = {
        'char_length_vs_word_count': df['sentence_char_length'].corr(df['actual_word_count']),
        'target_vs_actual_length': df['target_length'].corr(df['actual_word_count'])
    }
    
    # Distribution counts
    analysis['distribution_counts'] = {
        'by_language': df['language'].value_counts().to_dict(),
        'by_emotion': df['emotion'].value_counts().to_dict(),
        'by_industry': df['industry'].value_counts().to_dict(),
        'by_complex_words': df['use_complex_words'].value_counts().to_dict(),
        'by_scenario_type': df['scenario_type'].value_counts().to_dict(),
        'by_gender': df['gender'].value_counts().to_dict()
    }
    
    return analysis

def save_analysis_to_files(analysis: Dict, base_filename: str = "tts_analysis"):
    """Save comprehensive analysis to multiple files"""
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    
    # Convert numpy types to native Python types for JSON serialization
    def convert_numpy_types(obj):
        if isinstance(obj, dict):
            return {k: convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        elif hasattr(obj, 'item'):  # numpy scalar
            return obj.item()
        else:
            return obj
    
    # Save detailed JSON analysis
    json_filename = f"{base_filename}_{timestamp}.json"
    with open(json_filename, 'w', encoding='utf-8') as f:
        json.dump(convert_numpy_types(analysis), f, ensure_ascii=False, indent=2)
    
    # Save summary CSV for quick reference
    summary_data = {
        'metric': [],
        'value': []
    }
    
    # Add key metrics to summary
    summary_data['metric'].extend([
        'total_sentences', 'total_words', 'total_characters', 'estimated_speech_hours',
        'mean_char_length', 'median_char_length', 'std_char_length',
        'mean_word_count', 'median_word_count', 'std_word_count',
        'target_accuracy_rate', 'char_word_correlation'
    ])
    
    summary_data['value'].extend([
        analysis['total_sentences'],
        analysis['total_words'],
        analysis['total_characters'],
        round(analysis['estimated_speech_hours'], 4),
        round(analysis['character_length_distribution']['overall']['mean'], 2),
        round(analysis['character_length_distribution']['overall']['50%'], 2),
        round(analysis['character_length_distribution']['overall']['std'], 2),
        round(analysis['word_count_distribution']['overall']['mean'], 2),
        round(analysis['word_count_distribution']['overall']['50%'], 2),
        round(analysis['word_count_distribution']['overall']['std'], 2),
        round(analysis['target_vs_actual']['accuracy_rate'], 4),
        round(analysis['correlations']['char_length_vs_word_count'], 4)
    ])
    
    summary_df = pd.DataFrame(summary_data)
    summary_csv_filename = f"{base_filename}_summary_{timestamp}.csv"
    summary_df.to_csv(summary_csv_filename, index=False, encoding='utf-8')
    
    # Save detailed percentile analysis
    percentile_data = []
    
    # Character length percentiles by language
    for lang, stats in analysis['character_length_distribution']['by_language'].items():
        for percentile, value in stats.items():
            if percentile.endswith('%') or percentile in ['min', 'max']:
                percentile_data.append({
                    'category': 'character_length',
                    'subcategory': 'language',
                    'item': lang,
                    'percentile': percentile,
                    'value': value
                })
    
    # Word count percentiles by language
    for lang, stats in analysis['word_count_distribution']['by_language'].items():
        for percentile, value in stats.items():
            if percentile.endswith('%') or percentile in ['min', 'max']:
                percentile_data.append({
                    'category': 'word_count',
                    'subcategory': 'language',
                    'item': lang,
                    'percentile': percentile,
                    'value': value
                })
    
    percentile_df = pd.DataFrame(percentile_data)
    percentile_csv_filename = f"{base_filename}_percentiles_{timestamp}.csv"
    percentile_df.to_csv(percentile_csv_filename, index=False, encoding='utf-8')
    
    return {
        'json_file': json_filename,
        'summary_file': summary_csv_filename,
        'percentile_file': percentile_csv_filename
    }

class HourBasedSentenceGenerator:
    def __init__(self, hours_per_language: float = 1.0, sentences_per_call: int = N_SENTENCES_PER_CALL):
        self.hours_per_language = hours_per_language
        self.words_needed_per_language = int(WORDS_PER_HOUR * hours_per_language)
        self.sentences_per_call = sentences_per_call
        self.generated_sentences = []
        
        print(f"📊 Configuration:")
        print(f"   Hours per language: {hours_per_language}")
        print(f"   Words per hour: {WORDS_PER_HOUR}")
        print(f"   Target words per language: {self.words_needed_per_language}")
        print(f"   Sentences per GPT call: {sentences_per_call}")
        print(f"   🔤 English words will be transliterated into native scripts")
    
    def calculate_calls_needed(self, avg_sentence_length: float = 25) -> int:
        """Calculate number of GPT calls needed based on target word count"""
        total_sentences_needed = int(self.words_needed_per_language / avg_sentence_length)
        return int(total_sentences_needed / self.sentences_per_call) + 1
    
    async def generate_multiple_sentences(self, language: str, sentence_params: List[Dict]) -> List[Dict[str, Any]]:
        """Generate multiple sentences with specified parameters in one GPT call"""
        # Create instructions for each sentence
        sentence_instructions = []
        for i, params in enumerate(sentence_params, 1):
            customer_instruction = ""
            if params['customer_type'] == 'male_customer':
                customer_instruction = "Address as MALE customer"
            elif params['customer_type'] == 'female_customer':
                customer_instruction = "Address as FEMALE customer"
            else:
                customer_instruction = "Use NEUTRAL addressing (general respectful forms)"
            
            complex_instruction = "Include sophisticated industry-appropriate complex words" if params['use_complex'] else "Use regular vocabulary"
            
            sentence_instructions.append(f"""
Sentence {i}:
- Length: ~{params['target_length']} words
- Emotion: {params['emotion']}
- Industry: {params['industry']}
- Addressee: {customer_instruction}
- Complexity: {complex_instruction}""")
        
        system_prompt = f"""You are an expert at creating natural female-oriented code-switching sentences for Indian sales/service scenarios.

CRITICAL INSTRUCTIONS:

1. {FEMALE_CODESWITCHING_INSTRUCTIONS[language]}

2. All sentences MUST sound like they're spoken by a FEMALE professional in sales/service

3. Use warm, caring, professional tone typical of female customer service representatives

4. Generate EXACTLY {len(sentence_params)} sentences according to the specifications below

5. **CRUCIAL**: All English words/concepts MUST be transliterated into the native script. NO Roman script should appear in non-English languages.

SENTENCE SPECIFICATIONS:

{''.join(sentence_instructions)}

IMPORTANT: The SPEAKER is female but the CUSTOMER being addressed should vary as specified.

FEMALE SPEECH CHARACTERISTICS TO INCLUDE:

- Warm and empathetic tone
- Supportive and reassuring language
- Professional yet caring approach
- Use inclusive language
- Show active listening and concern
- Natural feminine speech patterns

INDUSTRY CONTEXTS:

- BFSI: Banking, insurance, financial services, loans, investments
- Healthcare: Medical services, wellness, patient care, pharmaceuticals
- Manufacturing: Production, quality control, supply chain, machinery
- Quick_Commerce: Fast delivery, e-commerce, logistics, warehousing
- EdTech: Education technology, online learning, curriculum, assessment
- Real_Estate: Property, housing, investments, mortgages, architecture
- Retail_Marketplace: Shopping, products, customer experience, inventory
- Automotive: Cars, service, parts, maintenance, dealerships
- Hospitality: Hotels, restaurants, travel, accommodation, tourism

OUTPUT FORMAT:

Return a JSON object with this structure:

{{
  "sentences": [
    "sentence 1 text",
    "sentence 2 text",
    ...
  ]
}}

Generate {len(sentence_params)} sentences that meet all criteria with properly transliterated English concepts."""

        user_prompt = f"""Generate {len(sentence_params)} {language} sentences for a FEMALE professional according to the specifications provided.

Each sentence should:
- Sound natural and caring
- Be professionally appropriate
- Follow the specific parameters for length, emotion, industry, and addressee type
- Use feminine speech patterns while addressing various customer types
- Have all English words/concepts transliterated into {language} script (NO Roman script)

Create sentences that sound authentic for female customer service representatives in Indian call centers, with English concepts properly transliterated into native script."""

        async with semaphore:
            try:
                response = await async_gpt4o_client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    model='gpt-4o',
                    temperature=0.8,
                )
                
                result = response.choices[0].message.content.strip()
                print(f"Generated response for {language}: {result[:200]}...")  # Print first 200 chars for debugging
                
                               # Remove `````` markers if present
                if result.startswith("```json"):
                    result = result.replace("```json", "").replace("```", "").strip()
                elif result.startswith("```"):
                    result = result.replace("```", "").strip()
                elif result.startswith("```plaintext"):
                    result = result.replace("``````", "").strip()
                elif result.startswith("```text"):
                    result = result.replace("```text", "").replace("```", "").strip()

                # parse between the first and last curly braces
                start_index = result.find('{')
                end_index = result.rfind('}')
                result = result[start_index:end_index + 1].strip()
                if not result:
                    print(f"Empty response for {language}. Check the prompt and parameters.")
                    return []
                if not result.startswith('{') or not result.endswith('}'):
                    print(f"Invalid JSON format for {language}. Response: {result[:200]}...")
                    return []
                
                # Try to parse JSON response
                try:
                    parsed_result = json.loads(result)
                    sentences_list = parsed_result.get('sentences', [])
                except json.JSONDecodeError:
                    print(f"JSON parse error for {language}. Raw response: {result[:200]}...")
                    return []
                
                # Process each sentence
                processed_sentences = []
                for i, sentence_text in enumerate(sentences_list):
                    if i >= len(sentence_params):  # Safety check
                        break
                    if not sentence_text or not sentence_text.strip():
                        continue
                    
                    # Calculate word count manually using split function
                    actual_word_count = calculate_word_count(sentence_text)
                    
                    processed_sentence = {
                        'language': language,
                        'generated_sentence': sentence_text,
                        'sentence_char_length': len(sentence_text),
                        'target_length': sentence_params[i]['target_length'],
                        'actual_word_count': actual_word_count,
                        'emotion': sentence_params[i]['emotion'],
                        'industry': sentence_params[i]['industry'],
                        'use_complex_words': sentence_params[i]['use_complex'],
                        'gender': 'female',
                        'scenario_type': 'sales_service',
                        'generation_id': f"{language}_{sentence_params[i]['emotion']}_{int(time.time())}_{random.randint(1000, 9999)}"
                    }
                    
                    processed_sentences.append(processed_sentence)
                
                return processed_sentences
                
            except Exception as e:
                print(f"Error generating sentences for {language}: {e}")
                return []
    
    async def generate_for_language(self, language: str) -> List[Dict[str, Any]]:
        """Generate sentences for a specific language until target word count is reached"""
        calls_needed = self.calculate_calls_needed()
        print(f"\n🎯 Generating for {language}")
        print(f"   Target words: {self.words_needed_per_language}")
        print(f"   Sentences per call: {self.sentences_per_call}")
        print(f"   Estimated GPT calls needed: {calls_needed}")
        
        results = []
        total_words = 0
        call_count = 0
        start_time = time.time()
        
        while total_words < self.words_needed_per_language:
            # Generate parameters for multiple sentences
            sentence_params = generate_sentence_parameters(self.sentences_per_call)
            
            # Generate multiple sentences in one call
            batch_results = await self.generate_multiple_sentences(language, sentence_params)
            # print(batch_results)
            
            if batch_results:
                results.extend(batch_results)
                batch_words = sum([result['actual_word_count'] for result in batch_results])
                total_words += batch_words
                call_count += 1
                
                # Progress update every 5 calls
                if call_count % 5 == 0:
                    elapsed = time.time() - start_time
                    words_per_sec = total_words / elapsed if elapsed > 0 else 0
                    remaining_words = self.words_needed_per_language - total_words
                    eta = remaining_words / words_per_sec if words_per_sec > 0 else 0
                    progress_pct = (total_words / self.words_needed_per_language) * 100
                    print(f"   {language} Progress: {call_count} calls, {len(results)} sentences, {total_words}/{self.words_needed_per_language} words ({progress_pct:.1f}%) | ETA: {eta:.1f}s")
            else:
                print(f"   Warning: No sentences generated in call {call_count + 1}")
            
            # Rate limiting delay
            await asyncio.sleep(0.2)  # Slightly longer delay since we're doing more per call
        
        elapsed = time.time() - start_time
        print(f"✅ {language} completed: {call_count} calls, {len(results)} sentences, {total_words} words in {elapsed:.1f}s")
        print(f"   Efficiency: {len(results)/call_count:.1f} sentences per call")
        print(f"   🔤 All English concepts transliterated to {language} script")
        
        return results
    
    def save_temp_csv(self, sentences: List[Dict[str, Any]], language: str):
        """Save temporary CSV for a specific language"""
        if not sentences:
            return False
        
        df = pd.DataFrame(sentences)
        temp_filename = f"{language.lower()}_sentences_transliterated.csv"
        
        # Only keep the requested columns
        column_order = [
            'language', 'generated_sentence', 'sentence_char_length', 'target_length',
            'actual_word_count', 'emotion', 'industry', 'use_complex_words',
            'gender', 'scenario_type', 'generation_id'
        ]
        
        df = df[column_order]
        df.to_csv(temp_filename, index=False, encoding='utf-8')
        print(f"💾 Temporary file saved: {temp_filename} ({len(df)} sentences)")
        return True
    
    def save_final_csv(self, all_sentences: List[Dict[str, Any]], filename: str = "generated_female_sentences_transliterated.csv"):
        """Save final combined CSV"""
        if not all_sentences:
            print("No sentences to save!")
            return False
        
        df = pd.DataFrame(all_sentences)
        
        # Only keep the requested columns
        column_order = [
            'language', 'generated_sentence', 'sentence_char_length', 'target_length',
            'actual_word_count', 'emotion', 'industry', 'use_complex_words',
            'gender', 'scenario_type', 'generation_id'
        ]
        
        df = df[column_order]
        df.to_csv(filename, index=False, encoding='utf-8')
        print(f"\n💾 Final dataset saved: {filename}")
        print(f"📊 Total sentences: {len(df)}")
        print(f"📊 Total words: {df['actual_word_count'].sum()}")
        print(f"📊 Total characters: {df['sentence_char_length'].sum()}")
        print(f"🔤 All English concepts properly transliterated to native scripts")
        
        return True
    
    # NEW METHOD: Generate single language as a task
    async def generate_language_task(self, language: str, task_id: int, total_languages: int) -> Dict[str, Any]:
        """Generate sentences for a single language as an async task"""
        print(f"\n🚀 Starting {language} generation (Task {task_id}/{total_languages}) - Transliterated English")
        try:
            sentences = await self.generate_for_language(language)
            if sentences:
                # Save temporary CSV for safety
                self.save_temp_csv(sentences, language)
                print(f"✅ {language} completed successfully with {len(sentences)} sentences (English transliterated)")
                return {
                    'language': language,
                    'sentences': sentences,
                    'success': True,
                    'error': None
                }
            else:
                print(f"❌ No sentences generated for {language}")
                return {
                    'language': language,
                    'sentences': [],
                    'success': False,
                    'error': 'No sentences generated'
                }
        except Exception as e:
            print(f"❌ Error processing {language}: {e}")
            return {
                'language': language,
                'sentences': [],
                'success': False,
                'error': str(e)
            }
    
    # MODIFIED METHOD: Generate all languages simultaneously
    async def generate_complete_dataset(self, languages: List[str] = None):
        """Generate complete dataset for all specified languages SIMULTANEOUSLY"""
        if languages is None:
            languages = LANGUAGES
        
        print(f"🎀 SIMULTANEOUS HOUR-BASED FEMALE SENTENCE GENERATION (TRANSLITERATED)")
        print("=" * 80)
        print(f"Configuration:")
        print(f"   Languages: {', '.join(languages)}")
        print(f"   Hours per language: {self.hours_per_language}")
        print(f"   Words per language: {self.words_needed_per_language}")
        print(f"   Total target words: {self.words_needed_per_language * len(languages)}")
        print(f"   Speech rate: {WORDS_PER_MINUTE} WPM")
        print(f"   Sentences per GPT call: {self.sentences_per_call}")
        print(f"   Word counting: Manual calculation using split() method")
        print(f"   Speaker: Female (consistent)")
        print(f"   Addressee: Variable (male/female/neutral customers)")
        print(f"   🔤 English concepts: Transliterated to native scripts (NO Roman script)")
        print(f"   🚀 CONCURRENT PROCESSING: All {len(languages)} languages will run simultaneously!")
        
        print(f"\n🏁 Starting simultaneous generation for all languages...")
        start_time = time.time()
        
        # Create tasks for all languages
        language_tasks = []
        for i, language in enumerate(languages, 1):
            task = asyncio.create_task(
                self.generate_language_task(language, i, len(languages))
            )
            language_tasks.append(task)
        
        # Run all language generation tasks concurrently
        print(f"⚡ Running {len(language_tasks)} language generation tasks concurrently...")
        results = await asyncio.gather(*language_tasks, return_exceptions=True)
        
        total_time = time.time() - start_time
        print(f"\n🏁 All language tasks completed in {total_time:.1f} seconds")
        
        # Process results
        all_sentences = []
        successful_languages = []
        failed_languages = []
        
        for result in results:
            if isinstance(result, Exception):
                print(f"❌ Task failed with exception: {result}")
                failed_languages.append(f"Unknown: {str(result)}")
            elif result.get('success', False):
                all_sentences.extend(result['sentences'])
                successful_languages.append(result['language'])
                print(f"✅ {result['language']}: {len(result['sentences'])} sentences (transliterated)")
            else:
                failed_languages.append(f"{result['language']}: {result.get('error', 'Unknown error')}")
                print(f"❌ {result['language']}: {result.get('error', 'Unknown error')}")
        
        # Summary of concurrent execution
        print(f"\n📊 CONCURRENT EXECUTION SUMMARY:")
        print(f"   Successful languages: {len(successful_languages)}")
        print(f"   Failed languages: {len(failed_languages)}")
        print(f"   Total execution time: {total_time:.1f} seconds")
        print(f"   Average time per language: {total_time/len(languages):.1f} seconds")
        print(f"   Time saved vs sequential: ~{max(0, (total_time * len(languages)) - total_time):.1f} seconds")
        
        if successful_languages:
            print(f"   Successful: {', '.join(successful_languages)}")
        if failed_languages:
            print(f"   Failed: {', '.join(failed_languages)}")
        
        if all_sentences:
            # Save final combined dataset
            success = self.save_final_csv(all_sentences, "generated_female_sentences_transliterated.csv")
            
            if success:
                # Create and save comprehensive analysis
                print(f"\n📊 Creating comprehensive distribution analysis...")
                analysis = create_comprehensive_analysis(all_sentences)
                analysis_files = save_analysis_to_files(analysis, "tts_analysis_transliterated")
                
                # Display key results
                self.display_analysis_summary(analysis)
                
                print(f"\n🎉 CONCURRENT GENERATION COMPLETED SUCCESSFULLY!")
                print(f"✅ Generated sentences for {len(successful_languages)} languages simultaneously")
                print(f"✅ Total sentences: {len(all_sentences)}")
                print(f"✅ Total words: {sum([s['actual_word_count'] for s in all_sentences])}")
                print(f"✅ Total time: {total_time:.1f} seconds (vs ~{total_time * len(languages):.1f}s sequential)")
                print(f"🔤 All English concepts properly transliterated to native scripts")
                
                print(f"\n📁 Analysis files saved:")
                for file_type, filename in analysis_files.items():
                    print(f"   {file_type}: {filename}")
                
                return all_sentences
        
        return []
    
    def display_analysis_summary(self, analysis: Dict):
        """Display summary of comprehensive analysis"""
        print(f"\n📊 COMPREHENSIVE DISTRIBUTION ANALYSIS (TRANSLITERATED)")
        print("=" * 70)
        
        print(f"\n🔢 Overall Statistics:")
        print(f"   Total sentences: {analysis['total_sentences']:,}")
        print(f"   Total words: {analysis['total_words']:,}")
        print(f"   Total characters: {analysis['total_characters']:,}")
        print(f"   Estimated speech time: {analysis['estimated_speech_hours']:.2f} hours")
        
        print(f"\n📏 Character Length Distribution:")
        char_overall = analysis['character_length_distribution']['overall']
        print(f"   Mean: {char_overall['mean']:.1f}")
        print(f"   Median (50%): {char_overall['50%']:.1f}")
        print(f"   90th percentile: {char_overall['90%']:.1f}")
        print(f"   95th percentile: {char_overall['95%']:.1f}")
        print(f"   Range: {char_overall['min']:.0f} - {char_overall['max']:.0f}")
        
        print(f"\n📝 Word Count Distribution:")
        word_overall = analysis['word_count_distribution']['overall']
        print(f"   Mean: {word_overall['mean']:.1f}")
        print(f"   Median (50%): {word_overall['50%']:.1f}")
        print(f"   90th percentile: {word_overall['90%']:.1f}")
        print(f"   95th percentile: {word_overall['95%']:.1f}")
        print(f"   Range: {word_overall['min']:.0f} - {word_overall['max']:.0f}")
        
        print(f"\n🎯 Target vs Actual Length:")
        target_analysis = analysis['target_vs_actual']
        print(f"   Mean difference: {target_analysis['mean_difference']:.2f} words")
        print(f"   Accuracy rate (±3 words): {target_analysis['accuracy_rate']:.1%}")
        print(f"   Accurate sentences: {target_analysis['accuracy_count']}")
        
        print(f"\n🔗 Correlations:")
        correlations = analysis['correlations']
        print(f"   Character length vs Word count: {correlations['char_length_vs_word_count']:.3f}")
        print(f"   Target vs Actual length: {correlations['target_vs_actual_length']:.3f}")
        
        print(f"\n🗂️ Distribution by Category:")
        counts = analysis['distribution_counts']
        print(f"   Languages: {len(counts['by_language'])} ({list(counts['by_language'].keys())})")
        print(f"   Emotions: {len(counts['by_emotion'])} (top 3: {list(sorted(counts['by_emotion'].items(), key=lambda x: x[1], reverse=True)[:3])})")
        print(f"   Industries: {len(counts['by_industry'])} (top 3: {list(sorted(counts['by_industry'].items(), key=lambda x: x[1], reverse=True)[:3])})")
        print(f"   Complex words usage: {counts['by_complex_words']}")

async def main():
    """Main function with configurable hours per language and sentences per call"""
    # Configuration - MODIFY THESE PARAMETERS
    HOURS_PER_LANGUAGE = N_HOURS_PER_LANGUAGE  # Hours of speech data per language
    SENTENCES_PER_CALL = N_SENTENCES_PER_CALL  # Number of sentences generated per GPT call
    LANGUAGES_TO_GENERATE = LANGUAGES  # Use all languages or specify subset: ['Hindi', 'English', 'Tamil']
    
    print("🎀 STARTING SIMULTANEOUS HOUR-BASED SENTENCE GENERATION (TRANSLITERATED)")
    print(f"Target: {HOURS_PER_LANGUAGE} hours per language")
    print(f"Sentences per GPT call: {SENTENCES_PER_CALL}")
    print("Word counting: Manual calculation using Python split() method")
    print("Analysis: Comprehensive percentile distributions with file export")
    print("Speaker: Female professional (consistent)")
    print("Addressee: Variable customer types (male/female/neutral)")
    print("🔤 English transliteration: GPT handles transliteration to native scripts")
    print("🚀 CONCURRENT PROCESSING: All languages run simultaneously!")
    
    # Initialize generator
    generator = HourBasedSentenceGenerator(
        hours_per_language=HOURS_PER_LANGUAGE,
        sentences_per_call=SENTENCES_PER_CALL
    )
    
    # Generate complete dataset with concurrent processing
    sentences = await generator.generate_complete_dataset(LANGUAGES_TO_GENERATE)
    
    if sentences:
        print(f"\n🎉 SUCCESS! Generated {len(sentences)} sentences concurrently")
        total_hours = sum([s['actual_word_count'] for s in sentences]) / WORDS_PER_HOUR
        print(f"📊 Total speech time: {total_hours:.2f} hours")
        print(f"🔤 All English concepts transliterated to native scripts by GPT")
    else:
        print("❌ Generation failed!")

# Run the script
if __name__ == "__main__":
    # For Jupyter compatibility
    try:
        import nest_asyncio
        nest_asyncio.apply()
    except ImportError:
        pass
    
    print("Starting Simultaneous Hour-Based Female Sentence Generation with Transliterated English...")
    asyncio.run(main())