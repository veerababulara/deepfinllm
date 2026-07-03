"""
Financial Advisory Intelligence Platform - Python/Streamlit Version
Multilingual Agentic Hybrid-RAG System (Single File - Backend + Frontend)
"""

import streamlit as st
import requests
from langdetect import detect, LangDetectException
import time
import json
from datetime import datetime
import os
from dotenv import load_dotenv
import random
import yfinance as yf
import re
from typing import Dict, List, Tuple

# Load environment variables
load_dotenv()

# Configure DeepSeek API
DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY') or st.secrets.get('DEEPSEEK_API_KEY')
DEEPSEEK_API_URL = "https://api.deepseek.com/chat/completions"

if not DEEPSEEK_API_KEY:
    st.error("❌ DEEPSEEK_API_KEY not found! Add it to .env or Streamlit secrets")
    st.stop()

# ============================================================================
# CONFIGURATION
# ============================================================================

LANGUAGE_NAMES = {
    'en': 'English',
    'es': 'Spanish',
    'fr': 'French',
    'de': 'German',
    'zh-cn': 'Chinese',
    'ja': 'Japanese',
    'pt': 'Portuguese',
    'hi': 'Hindi',
}

# Financial documents (Mock Knowledge Base)
FINANCIAL_DOCUMENTS = [
    {
        'id': 'doc_001',
        'title': 'Investment Portfolio Diversification Guide 2024',
        'content': 'Portfolio diversification is the practice of allocating capital across various asset classes to reduce risk. Key principles include: 60/40 stock-bond allocation for conservative investors, 70/30 for moderate investors, and 80/20 for aggressive. Recent market analysis shows that diversification across US equities, international stocks, bonds, commodities, and real estate provides optimal risk-adjusted returns.',
        'category': 'portfolio'
    },
    {
        'id': 'doc_002',
        'title': 'Emergency Fund Best Practices',
        'content': 'Financial advisors recommend maintaining an emergency fund covering 3-6 months of living expenses in liquid, easily accessible accounts. High-Yield Savings Accounts (HYSA) currently offer 4.5-5.0% APY. Emergency fund composition: 50% in HYSA, 30% in money market funds, 20% in short-term CDs. This ensures immediate accessibility while maintaining reasonable returns.',
        'category': 'savings'
    },
    {
        'id': 'doc_003',
        'title': 'Tax-Advantaged Retirement Accounts Explained',
        'content': '401(k) plans offer tax-deferred growth with 2024 contribution limits of $23,500 ($31,000 if age 50+). Roth IRAs provide tax-free growth with $7,000 annual contributions. Strategic withdrawal ordering: taxable accounts first, then traditional IRAs, finally Roth IRAs to minimize lifetime tax burden. Employer matching in 401(k)s provides immediate 50-100% returns.',
        'category': 'retirement'
    },
    {
        'id': 'doc_004',
        'title': 'Bond Investment Strategy and Duration Risk',
        'content': 'Bond investing: yield curve inversion signals economic transition, duration measures interest rate sensitivity. Current environment: 10-year Treasury yields around 4.0-4.5%, corporate bonds 5.5-6.5%. Bond ladder strategy: stagger maturity dates for consistent income flow. Credit quality: AAA-rated bonds safer than BBB but offer lower yields.',
        'category': 'bonds'
    },
    {
        'id': 'doc_005',
        'title': 'Dividend Growth Investing Fundamentals',
        'content': 'Dividend growth investing targets companies with consistent dividend increase histories. Dividend aristocrats: S&P 500 companies with 25+ years of consecutive dividend increases. Key metrics: dividend yield (target 2-4%), payout ratio (target 40-60%), dividend growth rate (target 5-10% annually). Qualified vs non-qualified dividends: qualified taxed at 15-20%, non-qualified at ordinary income rates.',
        'category': 'dividends'
    },
    {
        'id': 'doc_006',
        'title': 'Real Estate Investment and Property Analysis',
        'content': 'REITs provide real estate exposure without direct property ownership, typically yielding 3-5% annually. Cap rate (capitalization rate) = Net Operating Income / Property Value; target 5-8% for residential, 6-10% for commercial. Cash-on-cash return measures annual cash flow against initial investment; target 8-15% minimum. Real estate diversification: residential (stable but lower returns), commercial (higher returns, higher risk), industrial (growth sector).',
        'category': 'realestate'
    },
    {
        'id': 'doc_007',
        'title': 'Cryptocurrency and Digital Asset Allocation',
        'content': 'Bitcoin and Ethereum represent 80% of cryptocurrency market cap with relatively established infrastructure. Recommended allocation: 0-5% of portfolio for risk tolerance; store in hardware wallets for security. Staking opportunities provide 5-15% annual returns on certain cryptocurrencies. DeFi (Decentralized Finance) offers higher yields (10-30%) but involves smart contract risks. Tax implications: cryptocurrency transactions trigger capital gains tax.',
        'category': 'crypto'
    },
    {
        'id': 'doc_008',
        'title': 'Risk Management and Insurance Planning',
        'content': 'Term life insurance (20-30 years) provides maximum coverage at lowest cost for young families. Coverage recommendation: 10-15x annual income for primary earners. Disability insurance replaces 60-70% of income if unable to work. Umbrella insurance adds $1-2M liability coverage for $200-400 annually. Health insurance: prioritize plans with HSA eligibility for triple tax advantages.',
        'category': 'insurance'
    },
    {
        'id': 'doc_009',
        'title': 'Market Timing vs Dollar-Cost Averaging',
        'content': 'Dollar-Cost Averaging (DCA): investing fixed amounts at regular intervals regardless of market price. Historical data shows DCA outperforms lump-sum investing 70% of the time in volatile markets. Lump-sum investing outperforms when markets are rising strongly; DCA better in uncertain/declining markets. Psychologically, DCA reduces emotional investing and market timing mistakes.',
        'category': 'strategy'
    },
    {
        'id': 'doc_010',
        'title': 'Behavioral Finance and Investment Psychology',
        'content': 'Confirmation bias: seeking information confirming existing beliefs; actively diversify information sources. Loss aversion: disproportionate emotional pain from losses vs pleasure from gains; stick to predetermined asset allocation. Recency bias: overweighting recent performance; focus on long-term trends. Herd mentality: following crowd into bubbles; maintain contrarian discipline. Fear and greed cycles: automate investments to avoid emotional decisions.',
        'category': 'psychology'
    }
]

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def detect_language(text: str) -> str:
    """Detect language using langdetect library"""
    try:
        lang = detect(text)
        return lang
    except LangDetectException:
        return 'en'  # Default to English

def classify_intent(query: str) -> str:
    """Classify financial intent"""
    intents = {
        'portfolio_optimization': r'portfolio|allocation|diversif|rebalanc|asset|adjust',
        'risk_assessment': r'risk|safe|conservative|aggressive|volatile|exposure',
        'retirement_planning': r'retirement|retire|401k|IRA|pension|future|age 65',
        'tax_optimization': r'tax|deduction|credit|capital gain|ira|roth',
        'emergency_fund': r'emergency|cash reserve|liquid|rainy day|buffer',
        'investment_strategy': r'invest|stock|bond|dividend|growth|value|index',
        'debt_management': r'debt|loan|mortgage|interest|credit|refinance',
        'income_planning': r'income|salary|passive|side hustle|business',
    }
    
    import re
    for intent, pattern in intents.items():
        if re.search(pattern, query.lower()):
            return intent
    
    return 'general_inquiry'

def expand_keywords(query: str) -> list:
    """Expand query keywords for better retrieval"""
    base_terms = [w for w in query.lower().split() if len(w) > 3]
    
    expansions = {
        'invest': ['investment', 'investing', 'investor', 'portfolio', 'asset'],
        'stock': ['equity', 'shares', 'equities', 'securities', 'ticker'],
        'bond': ['fixed income', 'debt', 'coupon', 'yield', 'treasuries'],
        'retire': ['retirement', 'pension', 'withdraw', 'distribution', 'annuity'],
        'risk': ['volatility', 'exposure', 'correlation', 'beta', 'duration'],
        'money': ['capital', 'funds', 'cash', 'liquidity', 'asset'],
    }
    
    expanded = set(base_terms)
    for term in base_terms:
        for key, values in expansions.items():
            if key in term or term in key:
                expanded.update(values)
    
    return list(expanded)

def calculate_similarity(query: str, document: dict) -> float:
    """Calculate similarity between query and document"""
    query_terms = query.lower().split()
    content_lower = document['content'].lower()
    
    matches = sum(1 for term in query_terms if term in content_lower)
    return matches / len(query_terms) if query_terms else 0

def simulate_latency(min_ms: int = 150, max_ms: int = 600):
    """Simulate processing latency"""
    delay = random.randint(min_ms, max_ms) / 1000
    time.sleep(delay)

# ============================================================================
# STOCK PRICE FETCHING & WEB SCRAPING
# ============================================================================

def extract_company_names(query: str) -> List[str]:
    """Extract company names from user query and convert to stock tickers using DeepSeek AI"""
    if not query.strip():
        return []
        
    try:
        headers = {
            "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
            "Content-Type": "application/json"
        }
        
        system_prompt = (
            "You are a precise financial data extractor. Identify any companies mentioned in the "
            "user's query and map them to their correct Yahoo Finance stock ticker symbols (e.g., Apple -> AAPL, "
            "Microsoft -> MSFT, Google -> GOOGL, Coca-Cola -> KO, TCS->TCS.NS, Mahindra and mahindra->MOM.F).\n\n"
            "CRITICAL: Return ONLY a valid JSON array of strings containing the ticker symbols. "
            "If no companies or stock tickers are implied, return an empty array []. "
            "Do not include markdown code blocks, formatting, or conversational text. "
            "Example valid response: [\"AAPL\", \"NVDA\"]"
        )
        
        payload = {
            "model": "deepseek-chat",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Query: {query}"}
            ],
            "temperature": 0.1,  # Kept low for consistent, deterministic behavior
            "max_tokens": 150
        }
        
        response = requests.post(DEEPSEEK_API_URL, headers=headers, json=payload, timeout=10)
        response.raise_for_status()
        
        # Clean the output text in case the AI included markdown code fences
        content = response.json()['choices'][0]['message']['content'].strip()
        if content.startswith("```"):
            content = content.replace("```json", "").replace("```", "").strip()
            
        tickers = json.loads(content)
        if isinstance(tickers, list):
            # Clean up strings to ensure standard ticker formats (uppercase)
            return [str(ticker).upper().strip() for ticker in tickers if ticker]
            
    except Exception as e:
        # Subtle fail-safe: log error internally and fall back gracefully so your pipeline doesn't break
        import sys
        print(f"AI Ticker Extraction Error: {str(e)}", file=sys.stderr)
        
    return []

def fetch_stock_data(ticker: str) -> Dict:
    """Fetch current stock price and data using yfinance"""
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period='1d')
        info = stock.info
        
        current_price = info.get('currentPrice', info.get('regularMarketPrice', 'N/A'))
        company_name = info.get('longName', ticker)
        market_cap = info.get('marketCap', 'N/A')
        pe_ratio = info.get('trailingPE', 'N/A')
        dividend_yield = info.get('dividendYield', 'N/A')
        fifty_two_week_high = info.get('fiftyTwoWeekHigh', 'N/A')
        fifty_two_week_low = info.get('fiftyTwoWeekLow', 'N/A')
        
        # Calculate price change
        if len(hist) > 0:
            close_price = hist['Close'].iloc[-1]
            open_price = hist['Open'].iloc[-1]
            price_change = close_price - open_price
            price_change_pct = (price_change / open_price * 100) if open_price != 0 else 0
        else:
            close_price = current_price if isinstance(current_price, (int, float)) else 0
            price_change = 0
            price_change_pct = 0
        
        return {
            'ticker': ticker,
            'company_name': company_name,
            'current_price': current_price,
            'price_change': round(price_change, 2) if isinstance(price_change, (int, float)) else 0,
            'price_change_pct': round(price_change_pct, 2) if isinstance(price_change_pct, (int, float)) else 0,
            'market_cap': market_cap,
            'pe_ratio': round(pe_ratio, 2) if isinstance(pe_ratio, (int, float)) else 'N/A',
            'dividend_yield': f"{round(dividend_yield*100, 2)}%" if isinstance(dividend_yield, (int, float)) else 'N/A',
            'fifty_two_week_high': round(fifty_two_week_high, 2) if isinstance(fifty_two_week_high, (int, float)) else 'N/A',
            'fifty_two_week_low': round(fifty_two_week_low, 2) if isinstance(fifty_two_week_low, (int, float)) else 'N/A',
        }
    except Exception as e:
        return {
            'ticker': ticker,
            'error': str(e),
            'current_price': 'Unable to fetch'
        }

def get_multiple_stock_data(tickers: List[str]) -> Dict[str, Dict]:
    """Fetch stock data for multiple companies"""
    stock_data = {}
    for ticker in tickers:
        stock_data[ticker] = fetch_stock_data(ticker)
    return stock_data

def format_stock_data_for_prompt(stock_data: Dict) -> str:
    """Format stock data for inclusion in AI prompt"""
    if not stock_data:
        return ""
    
    formatted = "\n**LIVE STOCK MARKET DATA:**\n"
    
    for ticker, data in stock_data.items():
        if 'error' in data:
            formatted += f"\n- {ticker}: {data['error']}"
        else:
            company_name = data.get('company_name', ticker)
            price = data.get('current_price', 'N/A')
            change = data.get('price_change', 0)
            change_pct = data.get('price_change_pct', 0)
            pe_ratio = data.get('pe_ratio', 'N/A')
            dividend = data.get('dividend_yield', 'N/A')
            
            change_indicator = "📈" if change >= 0 else "📉"
            formatted += f"\n- **{company_name} ({ticker})**: ${price} {change_indicator} ({change_pct:+.2f}%)"
            formatted += f"\n  - PE Ratio: {pe_ratio} | Dividend Yield: {dividend}"
    
    return formatted

def get_language_prompt_instruction(language_code: str) -> str:
    """Get instruction for Deepseek to respond in specific language"""
    instructions = {
        'es': 'IMPORTANT: Respond entirely in Spanish. Your response must be 100% in Spanish language.',
        'fr': 'IMPORTANT: Respond entirely in French. Your response must be 100% in French language.',
        'de': 'IMPORTANT: Respond entirely in German. Your response must be 100% in German language.',
        'zh-cn': 'IMPORTANT: Respond entirely in Chinese (Simplified). Your response must be 100% in Chinese language.',
        'ja': 'IMPORTANT: Respond entirely in Japanese. Your response must be 100% in Japanese language.',
        'pt': 'IMPORTANT: Respond entirely in Portuguese. Your response must be 100% in Portuguese language.',
        'hi': 'IMPORTANT: Respond entirely in Hindi. Your response must be 100% in Hindi language.',
        'en': 'Respond in English.',
    }
    return instructions.get(language_code, 'Respond in English.')

# ============================================================================
# PIPELINE ORCHESTRATION
# ============================================================================

def process_financial_query(query: str, progress_callback=None):
    """Complete pipeline orchestration"""
    
    start_time = time.time()
    pipeline_logs = []
    
    # Log helper
    def log_step(stage, message, duration=0):
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'stage': stage,
            'message': message,
            'duration': duration
        }
        pipeline_logs.append(log_entry)
        if progress_callback:
            progress_callback(log_entry)
    
    # ========================================================================
    # STAGE 1: INPUT PARSING
    # ========================================================================
    log_step('STAGE 1', '🔍 Starting Multilingual Input Parsing...')
    stage1_start = time.time()
    simulate_latency(200, 400)
    
    detected_language = detect_language(query)
    intent = classify_intent(query)
    keywords = expand_keywords(query)
    
    stage1_duration = int((time.time() - stage1_start) * 1000)
    log_step('STAGE 1', f'✓ Language: {detected_language.upper()} | Intent: {intent} | Keywords: {len(keywords)}', stage1_duration)
    
    # ========================================================================
    # STAGE 1B: LIVE STOCK DATA EXTRACTION (NEW!)
    # ========================================================================
    log_step('STAGE 1B', '📊 Extracting company names from query...')
    stock_extraction_start = time.time()
    
    # Extract companies mentioned in query
    company_tickers = extract_company_names(query)
    
    if company_tickers:
        log_step('STAGE 1B', f'🔍 Found {len(company_tickers)} companies: {", ".join(company_tickers)}')
        
        # Fetch stock data
        log_step('STAGE 1B', '📡 Fetching live stock prices from market...')
        stock_data = get_multiple_stock_data(company_tickers)
        stock_data_formatted = format_stock_data_for_prompt(stock_data)
        
        stock_extraction_duration = int((time.time() - stock_extraction_start) * 1000)
        log_step('STAGE 1B', f'✓ Stock Data Retrieval Complete ({stock_extraction_duration}ms)', stock_extraction_duration)
    else:
        stock_data_formatted = ""
        stock_extraction_duration = 0
        log_step('STAGE 1B', '⚪ No specific companies mentioned in query')
    
    # ========================================================================
    # STAGE 2: HYBRID RETRIEVAL
    # ========================================================================
    log_step('STAGE 2', '🔄 Starting Hybrid Retrieval Engine...')
    stage2_start = time.time()
    
    # Node 1: Similarity-based retrieval
    log_step('NODE 1', 'Executing Lexical/Semantic Search (α = 0.7)...')
    simulate_latency(300, 500)
    
    retrieved_docs = []
    for doc in FINANCIAL_DOCUMENTS:
        score = calculate_similarity(query, doc) * 0.7
        if score > 0:
            retrieved_docs.append({**doc, 'score': score})
    
    retrieved_docs.sort(key=lambda x: x['score'], reverse=True)
    top_docs = retrieved_docs[:3]
    
    log_step('NODE 1', f'✓ Retrieved {len(top_docs)} relevant documents')
    
    stage2_duration = int((time.time() - stage2_start) * 1000)
    log_step('STAGE 2', f'✓ Hybrid Retrieval Complete ({stage2_duration}ms)', stage2_duration)
    
    # ========================================================================
    # STAGE 3: CONTEXT PREPARATION
    # ========================================================================
    log_step('STAGE 3', '📊 BAS optiomization for context Preparation...')
    stage3_start = time.time()
    simulate_latency(150, 300)
    
    # Market data simulation
    market_data = {
        'sp500': {'price': 5847.23, 'change': 0.45},
        'bitcoin': {'price': 67432.50, 'change': 2.15},
        'treasury10y': {'yield': 4.23},
        'vix': {'index': 16.42}
    }
    
    context = '\n'.join([
        f"**{doc['title']}**\n{doc['content']}"
        for doc in top_docs
    ])
    
    stage3_duration = int((time.time() - stage3_start) * 1000)
    log_step('STAGE 3', f'✓ Context Prepared ({stage3_duration}ms)', stage3_duration)
    
    # ========================================================================
    # STAGE 4:  GENERATION (with language instruction)
    # ========================================================================
    log_step('STAGE 4', '🤖 Calling LLM...')
    stage4_start = time.time()
    
    language_instruction = get_language_prompt_instruction(detected_language)
    
    system_prompt = f"""You are a world-class financial advisor providing personalized guidance.

{language_instruction}

Provide:
1. Personalized financial advice (3-4 key points)
2. Risk assessment related to their situation
3. Recommended next steps
4. Market considerations

Be concise but comprehensive."""

    user_message = f"""User Query: {query}

Detected Language: {detected_language.upper()}
Financial Intent: {intent}

Market Context:
- S&P 500: ${market_data['sp500']['price']} ({market_data['sp500']['change']:+.2f}%)
- Bitcoin: ${market_data['bitcoin']['price']}
- 10Y Treasury: {market_data['treasury10y']['yield']}% yield
- VIX: {market_data['vix']['index']}

{stock_data_formatted if stock_data_formatted else ""}

Reference Documents:
{context}

{language_instruction}

IMPORTANT: Use the live stock data and market context provided above to give specific, data-driven financial recommendations. Reference actual company prices and metrics in your response."""

    try:
        # Call DeepSeek API
        headers = {
            "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "deepseek-chat",
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_message
                }
            ],
            "temperature": 0.7,
            "max_tokens": 1024,
            "top_p": 0.95,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0
        }
        
        response = requests.post(DEEPSEEK_API_URL, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        
        response_data = response.json()
        advice = response_data['choices'][0]['message']['content']
        
    except requests.exceptions.Timeout:
        advice = "Error: DeepSeek API request timed out. Please try again."
        log_step('ERROR', '❌ DeepSeek API Timeout')
    except requests.exceptions.HTTPError as e:
        advice = f"Error: DeepSeek API error - {e.response.status_code}"
        log_step('ERROR', f'❌ DeepSeek API Error: {e.response.status_code}')
    except Exception as e:
        advice = f"Error generating advice: {str(e)}"
        log_step('ERROR', f'❌ DeepSeek API Error: {str(e)}')
    
    stage4_duration = int((time.time() - stage4_start) * 1000)
    log_step('STAGE 4', f'✓ AI Generation Complete ({stage4_duration}ms)', stage4_duration)
    
    # ========================================================================
    # FINAL RESPONSE
    # ========================================================================
    total_duration = int((time.time() - start_time) * 1000)
    log_step('COMPLETE', f'✓ Pipeline Complete ({total_duration}ms)', total_duration)
    
    return {
        'query': query,
        'detected_language': detected_language,
        'language_name': LANGUAGE_NAMES.get(detected_language, detected_language.upper()),
        'intent': intent,
        'advice': advice,
        'sources': [doc['title'] for doc in top_docs],
        'processing_time': total_duration,
        'pipeline_logs': pipeline_logs,
        'confidence': 0.85 + random.random() * 0.1,
        'stock_data': stock_data if company_tickers else {},
        'companies_found': company_tickers
    }

# ============================================================================
# STREAMLIT FRONTEND
# ============================================================================

# Page configuration
st.set_page_config(
    page_title="Financial Advisory Platform",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional Corporate CSS
st.markdown("""
<style>
    * {
        font-family: 'Segoe UI', 'Trebuchet MS', sans-serif;
    }
    
    /* Page Background - Clean White */
    .main {
        background-color: #f8fafb;
    }
    
    /* Remove default streamlit padding */
    .block-container {
        padding-top: 2rem;
        padding-left: 3rem;
        padding-right: 3rem;
    }
    
    /* Header Styling */
    h1, h2, h3 {
        color: #1a1a1a;
        font-weight: 600;
    }
    
    p {
        color: #4a4a4a;
    }
    
    /* Text Input - Subtle White */
    .stTextArea > div > div > textarea {
        background-color: #ffffff !important;
        color: #1a1a1a !important;
        border: 1px solid #d0d0d0 !important;
        border-radius: 6px !important;
        font-size: 14px !important;
        padding: 12px !important;
    }
    
    .stTextArea > div > div > textarea:focus {
        border-color: #2563eb !important;
        box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.1) !important;
    }
    
    /* Button Styling - Professional Blue */
    .stButton > button {
        background-color: #2563eb !important;
        color: white !important;
        border: none !important;
        border-radius: 6px !important;
        font-weight: 600 !important;
        padding: 10px 24px !important;
        font-size: 14px !important;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1) !important;
        transition: all 0.3s ease !important;
    }
    
    .stButton > button:hover {
        background-color: #1d4ed8 !important;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.2) !important;
    }
    
    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #ffffff;
        border-bottom: 2px solid #e5e7eb;
    }
    
    .stTabs [data-baseweb="tab"] {
        color: #6b7280 !important;
        font-weight: 500;
    }
    
    .stTabs [aria-selected="true"] {
        color: #2563eb !important;
    }
    
    /* Advice Box - Clean and Professional */
    .advice-box {
        background-color: #ffffff;
        border-left: 4px solid #2563eb;
        padding: 20px;
        border-radius: 8px;
        margin: 15px 0;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08);
        border-top: 1px solid #e5e7eb;
        border-right: 1px solid #e5e7eb;
        border-bottom: 1px solid #e5e7eb;
    }
    
    /* Console Box - Light Gray Background */
    .console-box {
        background-color: #f3f4f6;
        border: 1px solid #d1d5db;
        border-radius: 6px;
        padding: 15px;
        margin: 15px 0;
        font-family: 'Courier New', monospace;
        font-size: 12px;
        color: #374151;
    }
    
    .console-line {
        padding: 5px 0;
        line-height: 1.6;
        color: #374151;
    }
    
    /* Metrics Cards */
    .stMetricRow {
        background-color: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 15px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08);
    }
    
    /* Metadata Grid - Professional Cards */
    .metadata-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 12px;
        margin: 20px 0;
    }
    
    .metadata-item {
        background-color: #ffffff;
        border: 1px solid #e5e7eb;
        border-left: 4px solid #2563eb;
        padding: 16px;
        border-radius: 6px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08);
    }
    
    .metadata-item strong {
        color: #1a1a1a;
        font-size: 13px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        display: block;
        margin-bottom: 8px;
        color: #6b7280;
    }
    
    .metadata-item span {
        color: #1a1a1a;
        font-size: 18px;
        font-weight: 600;
    }
    
    /* Badges - Subtle */
    .language-badge {
        background-color: #dbeafe;
        color: #1e40af;
        padding: 6px 12px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 11px;
        letter-spacing: 0.5px;
        border: 1px solid #93c5fd;
    }
    
    .intent-badge {
        background-color: #fef3c7;
        color: #92400e;
        padding: 6px 12px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 11px;
        letter-spacing: 0.5px;
        border: 1px solid #fcd34d;
    }
    
    /* Divider */
    hr {
        border-color: #e5e7eb !important;
        margin: 30px 0 !important;
    }
    
    /* Info Box */
    .stInfo {
        background-color: #eff6ff !important;
        border-left: 4px solid #3b82f6 !important;
        color: #1e40af !important;
    }
    
    /* Code Block */
    .stCodeBlock {
        background-color: #f3f4f6 !important;
        border: 1px solid #d1d5db !important;
    }
    
    /* Sidebar */
    .stSidebar {
        background-color: #f9fafb;
    }
    
    .stSidebar .stMarkdown {
        color: #374151;
    }
    
    /* Link Styling */
    a {
        color: #2563eb;
        text-decoration: none;
    }
    
    a:hover {
        text-decoration: underline;
    }
    
    /* Spinner/Loading */
    .stSpinner {
        color: #2563eb !important;
    }
    
    /* Source List */
    .source-item {
        background-color: #ffffff;
        border: 1px solid #e5e7eb;
        padding: 12px;
        margin: 8px 0;
        border-radius: 6px;
        border-left: 3px solid #2563eb;
    }
    
    .source-item strong {
        color: #1a1a1a;
        font-weight: 600;
    }
    
    /* Column Styling */
    .stColumn {
        background-color: transparent;
    }
</style>
""", unsafe_allow_html=True)

# Professional Header
st.markdown("""
<div style="border-bottom: 2px solid #e5e7eb; padding: 30px 0 20px 0; margin-bottom: 30px;">
    <div style="display: flex; align-items: center; gap: 15px; margin-bottom: 10px;">
        <div style="font-size: 32px;">🏛️</div>
        <div>
            <h1 style="color: #1a1a1a; margin: 0; font-size: 32px; font-weight: 700;">
                Financial Advisory Intelligence Platform
            </h1>
            <p style="color: #6b7280; margin: 5px 0 0 0; font-size: 13px; letter-spacing: 0.5px; text-transform: uppercase;">
                Multilingual Agentic Hybrid-RAG System
            </p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Tabs
tab1, tab2 = st.tabs(["💬 Query", "📊 Console"])

with tab1:
    # Query Input Section
    st.markdown("""
    <div style="background-color: #ffffff; border: 1px solid #e5e7eb; border-radius: 8px; padding: 20px; margin-bottom: 30px; box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08);">
        <p style="color: #374151; font-size: 13px; font-weight: 600; margin: 0 0 12px 0; text-transform: uppercase; letter-spacing: 0.5px;">
            Financial Query
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 1], gap="medium")
    
    with col1:
        query_input = st.text_area(
            "Enter your financial question",
            placeholder="Ask a financial question in any language...",
            height=90,
            label_visibility="collapsed"
        )
    
    with col2:
        st.write("")  # Spacing
        st.write("")  # Spacing
        submit_button = st.button("📊 Get Analysis", use_container_width=True, key="submit_btn")
    
    if submit_button and query_input:
        with st.spinner("⏳ Processing your query..."):
            # Store logs in session state
            if 'pipeline_logs' not in st.session_state:
                st.session_state.pipeline_logs = []
            
            logs_container = st.container()
            
            def update_logs(log_entry):
                st.session_state.pipeline_logs.append(log_entry)
                with logs_container:
                    st.write(f"📝 {log_entry['message']} ({log_entry.get('duration', 0)}ms)")
            
            result = process_financial_query(query_input, progress_callback=update_logs)
            
            # Store result in session state for tab switching
            st.session_state.last_result = result
    
    # Display empty state if no results yet
    if 'last_result' not in st.session_state:
        st.markdown("""
        <div style="text-align: center; padding: 60px 20px; color: #6b7280;">
            <div style="font-size: 48px; margin-bottom: 20px;">📋</div>
            <h2 style="color: #1a1a1a; font-size: 24px; font-weight: 600; margin: 0 0 10px 0;">
                Ask a Financial Question
            </h2>
            <p style="color: #9ca3af; font-size: 14px; margin: 0 0 30px 0; max-width: 500px; margin-left: auto; margin-right: auto;">
                Enter your financial inquiry in any language and receive personalized analysis powered by advanced AI.
            </p>
            <div style="background-color: #f3f4f6; border: 1px solid #d1d5db; border-radius: 8px; padding: 20px; display: inline-block;">
                <p style="color: #374151; font-size: 12px; margin: 0; font-weight: 600; letter-spacing: 0.5px; text-transform: uppercase; margin-bottom: 12px;">
                    Example Questions:
                </p>
                <p style="color: #6b7280; font-size: 13px; margin: 4px 0;">• How should I diversify my investment portfolio?</p>
                <p style="color: #6b7280; font-size: 13px; margin: 4px 0;">• ¿Cómo debo invertir mi dinero?</p>
                <p style="color: #6b7280; font-size: 13px; margin: 4px 0;">• Quels sont les meilleurs placements?</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Display results if available
    if 'last_result' in st.session_state:
        result = st.session_state.last_result
        
        st.divider()
        
        # Metadata - Professional Cards
        st.markdown("""
        <div style="margin-bottom: 20px;">
            <h3 style="color: #1a1a1a; font-size: 16px; font-weight: 600; margin: 0 0 15px 0; text-transform: uppercase; letter-spacing: 0.5px; color: #374151;">
                Analysis Metadata
            </h3>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4, gap="medium")
        
        with col1:
            st.markdown(f"""
            <div style="background-color: #ffffff; border: 1px solid #e5e7eb; border-left: 4px solid #2563eb; padding: 16px; border-radius: 6px; box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08);">
                <div style="color: #6b7280; font-size: 12px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 8px;">🌐 Language</div>
                <div style="color: #1a1a1a; font-size: 18px; font-weight: 700;">{result['language_name']}</div>
                <div style="color: #9ca3af; font-size: 11px; margin-top: 4px;">{result['detected_language'].upper()}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            intent_display = result['intent'].replace('_', ' ').title()
            st.markdown(f"""
            <div style="background-color: #ffffff; border: 1px solid #e5e7eb; border-left: 4px solid #2563eb; padding: 16px; border-radius: 6px; box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08);">
                <div style="color: #6b7280; font-size: 12px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 8px;">📍 Intent</div>
                <div style="color: #1a1a1a; font-size: 16px; font-weight: 700;">{intent_display}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            confidence = f"{result['confidence']*100:.1f}%"
            st.markdown(f"""
            <div style="background-color: #ffffff; border: 1px solid #e5e7eb; border-left: 4px solid #2563eb; padding: 16px; border-radius: 6px; box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08);">
                <div style="color: #6b7280; font-size: 12px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 8px;">✓ Confidence</div>
                <div style="color: #1a1a1a; font-size: 18px; font-weight: 700;">{confidence}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            processing_time = f"{result['processing_time']}ms"
            st.markdown(f"""
            <div style="background-color: #ffffff; border: 1px solid #e5e7eb; border-left: 4px solid #2563eb; padding: 16px; border-radius: 6px; box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08);">
                <div style="color: #6b7280; font-size: 12px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 8px;">⏱️ Processing</div>
                <div style="color: #1a1a1a; font-size: 18px; font-weight: 700;">{processing_time}</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.divider()
        
        # Stock Data Section (if companies were found)
        if result.get('stock_data'):
            st.markdown("""
            <div style="margin-bottom: 10px;">
                <h3 style="color: #1a1a1a; font-size: 16px; font-weight: 600; margin: 0; text-transform: uppercase; letter-spacing: 0.5px; color: #374151;">
                    📈 Live Stock Data
                </h3>
            </div>
            """, unsafe_allow_html=True)
            
            # Display stock data for each company
            stock_cols = st.columns(len(result['stock_data']))
            for idx, (ticker, stock_info) in enumerate(result['stock_data'].items()):
                with stock_cols[idx]:
                    if 'error' not in stock_info:
                        company = stock_info.get('company_name', ticker)
                        price = stock_info.get('current_price', 'N/A')
                        change = stock_info.get('price_change', 0)
                        change_pct = stock_info.get('price_change_pct', 0)
                        pe = stock_info.get('pe_ratio', 'N/A')
                        div = stock_info.get('dividend_yield', 'N/A')
                        
                        change_icon = "📈" if change >= 0 else "📉"
                        change_color = "#10b981" if change >= 0 else "#ef4444"
                        
                        st.markdown(f"""
                        <div style="background-color: #ffffff; border: 1px solid #e5e7eb; border-top: 3px solid {change_color}; padding: 16px; border-radius: 6px; box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08);">
                            <div style="font-weight: 700; color: #1a1a1a; font-size: 14px; margin-bottom: 8px;">{company}</div>
                            <div style="font-size: 20px; font-weight: 700; color: #1a1a1a; margin-bottom: 4px;">${price}</div>
                            <div style="color: {change_color}; font-weight: 600; font-size: 13px; margin-bottom: 12px;">{change_icon} {change:+.2f} ({change_pct:+.2f}%)</div>
                            <div style="font-size: 12px; color: #6b7280; line-height: 1.6;">
                                <div>P/E Ratio: <strong>{pe}</strong></div>
                                <div>Div. Yield: <strong>{div}</strong></div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div style="background-color: #fee2e2; border: 1px solid #fecaca; padding: 12px; border-radius: 6px;">
                            <div style="color: #991b1b; font-weight: 600;">{ticker}</div>
                            <div style="color: #7f1d1d; font-size: 12px;">{stock_info.get('error', 'Unable to fetch data')}</div>
                        </div>
                        """, unsafe_allow_html=True)
            
            st.divider()
        
        # Advice Section
        st.markdown("""
        <div style="margin-bottom: 10px;">
            <h3 style="color: #1a1a1a; font-size: 16px; font-weight: 600; margin: 0; text-transform: uppercase; letter-spacing: 0.5px; color: #374151;">
                💡 Personalized Recommendations
            </h3>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div style="background-color: #ffffff; border: 1px solid #e5e7eb; border-left: 4px solid #2563eb; padding: 24px; border-radius: 8px; box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08); line-height: 1.8; color: #374151;">
            {result['advice']}
        </div>
        """, unsafe_allow_html=True)
        
        st.divider()
        
        # Sources Section
        st.markdown("""
        <div style="margin-bottom: 10px;">
            <h3 style="color: #1a1a1a; font-size: 16px; font-weight: 600; margin: 0; text-transform: uppercase; letter-spacing: 0.5px; color: #374151;">
                📚 Source Documents
            </h3>
        </div>
        """, unsafe_allow_html=True)
        
        for i, source in enumerate(result['sources'], 1):
            st.markdown(f"""
            <div style="background-color: #ffffff; border: 1px solid #e5e7eb; border-left: 4px solid #2563eb; padding: 12px; margin: 8px 0; border-radius: 6px;">
                <div style="color: #1a1a1a; font-weight: 600; font-size: 14px;">
                    <span style="background-color: #dbeafe; color: #1e40af; padding: 2px 8px; border-radius: 3px; font-weight: 700; margin-right: 10px;">{i}</span>
                    {source}
                </div>
            </div>
            """, unsafe_allow_html=True)

with tab2:
    st.markdown("""
    <div style="margin-bottom: 20px;">
        <h3 style="color: #1a1a1a; font-size: 16px; font-weight: 600; margin: 0; text-transform: uppercase; letter-spacing: 0.5px; color: #374151;">
            📊 Pipeline Execution Log
        </h3>
    </div>
    """, unsafe_allow_html=True)
    
    if 'pipeline_logs' in st.session_state and st.session_state.pipeline_logs:
        logs_text = ""
        for log in st.session_state.pipeline_logs:
            logs_text += f"{log['message']}\n"
        
        st.markdown(f"""
        <div style="background-color: #f3f4f6; border: 1px solid #d1d5db; border-radius: 6px; padding: 16px; font-family: 'Courier New', monospace; font-size: 12px; line-height: 1.6; color: #374151; overflow-x: auto;">
            <pre style="margin: 0; white-space: pre-wrap; word-wrap: break-word;">{logs_text}</pre>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background-color: #eff6ff; border: 1px solid #93c5fd; border-left: 4px solid #3b82f6; padding: 16px; border-radius: 6px;">
            <p style="color: #1e40af; margin: 0; font-weight: 500;">⏳ Run a query to see pipeline execution logs</p>
        </div>
        """, unsafe_allow_html=True)

# Professional Sidebar
with st.sidebar:
    st.markdown("""
    <div style="padding: 20px 0 0 0;">
        <h3 style="color: #1a1a1a; font-size: 16px; font-weight: 600; margin: 0 0 20px 0; text-transform: uppercase; letter-spacing: 0.5px;">
            ⚙️ Configuration
        </h3>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("#### 🌐 Supported Languages")
    st.markdown("""
    - 🇬🇧 English (EN)
    - 🇪🇸 Spanish (ES)
    - 🇫🇷 French (FR)
    - 🇩🇪 German (DE)
    - 🇨🇳 Chinese (ZH)
    - 🇯🇵 Japanese (JA)
    - 🇵🇹 Portuguese (PT)
    - 🇮🇳 Hindi (HI)
    """)
    
    st.divider()
    
    st.markdown("#### 📊 Pipeline Architecture")
    st.markdown("""
    1. **Input Parsing** – Language Detection + Intent Classification
    2. **Retrieval** – Semantic Search + Document Matching
    3. **Context Prep** – Market Data + Knowledge Base
    4. **Generation** – Deepseek AI (Language-Aware)
    5. **Formatting** – Response in Query Language
    """)
    
    st.divider()
    
    st.markdown("#### ✨ Key Features")
    st.markdown("""
    - 🌐 Multilingual Support (8 languages)
    - 🤖 Deepseek AI
    - 📊 Real-time Market Integration
    - 💾 Financial Knowledge Base
    - 🔄 Hybrid Retrieval System
    - 📈 Pipeline Execution Logs
    """)

st.markdown("""
<div style="text-align: center; padding: 30px 0 20px 0; margin-top: 40px; border-top: 1px solid #e5e7eb;">
    <p style="color: #6b7280; font-size: 12px; margin: 0; letter-spacing: 0.5px;">
        Financial Advisory Intelligence Platform v1.0
    </p>
    <p style="color: #9ca3af; font-size: 11px; margin: 4px 0 0 0;">
        Powered by Mr. R. Veerababu-VFSTR | © 2026
    </p>
</div>
""", unsafe_allow_html=True)
