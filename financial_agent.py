from dotenv import load_dotenv
import os
import yfinance as yf
from duckduckgo_search import DDGS
from groq import Groq

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL = "llama-3.1-8b-instant"

def get_financial_data(ticker: str) -> dict:
    stock = yf.Ticker(ticker)

    info = stock.info
    recommendations = stock.recommendations

    analyst_summary = "No analyst data available."
    if recommendations is not None:
        analyst_summary = recommendations.tail(5).to_string()

    return {
        "company": info.get("longName", ticker),
        "current_price": info.get("currentPrice"),
        "market_cap": info.get("marketCap"),
        "analyst_recommendations": analyst_summary
    }

def get_latest_news(query: str, max_results=5):
    news = []
    with DDGS() as ddgs:
        for r in ddgs.news(query, max_results=max_results):
            news.append({
                "title": r["title"],
                "source": r["source"],
                "url": r["url"]
            })
    return news

def ask_llm(context: str):
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a financial analyst AI. "
                    "Summarize analyst sentiment and recent news clearly. "
                    "Use tables where useful."
                )
            },
            {
                "role": "user",
                "content": context
            }
        ],
        temperature=0.3
    )

    return response.choices[0].message.content

def run_financial_agent(ticker: str):
    finance_data = get_financial_data(ticker)
    news = get_latest_news(f"{ticker} stock news")

    context = f"""
Company: {finance_data['company']}
Current Price: {finance_data['current_price']}
Market Cap: {finance_data['market_cap']}

Analyst Recommendations:
{finance_data['analyst_recommendations']}

Latest News:
{news}
"""

    return ask_llm(context)

if __name__ == "__main__":
    output = run_financial_agent("NVDA")
    print(output)

