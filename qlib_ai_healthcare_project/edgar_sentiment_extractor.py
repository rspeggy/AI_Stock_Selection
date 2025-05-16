import os
import requests
import pandas as pd
import glob
import re
from sec_edgar_downloader import Downloader
from custom_lm_classifier import CustomLMClassifier

# 配置
industry_groups = {
    'AI Healthcare': ["GH", "EXAS", "ILMN", "TDOC", "MDT"],
    'Fintech': ["PYPL", "COIN", "AFRM", "SOFI", "UPST"],
    'Clean Energy': ["TSLA", "ENPH", "FSLR", "PLUG", "NEE"],
    'Cloud and Big Data': ["AMZN", "MSFT", "GOOGL", "SNOW", "CRM"],
    'Semiconductor': ["NVDA", "AMD", "INTC", "ASML", "TSM"]
}

REPORT_TYPES = ["10-K", "10-Q"]
DATE_RANGE = (2019, 2024)

# 文件夹
OUTPUT_DIR = "data/edgar"
DICTIONARY_PATH = "./Loughran-McDonald_MasterDictionary_1993-2024.csv"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# SEC EDGAR 配置
SEC_EMAIL = "peng198479@gmail.com"  # 替换为您的邮箱地址

# 初始化 SEC 下载器
downloader = Downloader(OUTPUT_DIR, SEC_EMAIL, download_folder=OUTPUT_DIR)

# 初始化情感分类器
lm_classifier = CustomLMClassifier(dictionary_path=DICTIONARY_PATH)

def remove_sec_header(text):
    """移除SEC文件头部信息"""
    print("\n=== 开始处理SEC文件头部 ===")
    print(f"原始文本长度: {len(text)}")
    
    # 查找第一个实际内容的位置
    content_start = text.find('<DOCUMENT>')
    if content_start == -1:
        content_start = text.find('<TEXT>')
    if content_start == -1:
        content_start = 0
    
    print(f"内容起始位置: {content_start}")
    
    # 提取实际内容
    text = text[content_start:]
    print(f"移除头部后长度: {len(text)}")
    
    # 移除常见的SEC文件头部标记，但保留内容
    patterns = [
        (r'<DOCUMENT>', ''),
        (r'</DOCUMENT>', ''),
        (r'<TEXT>', ''),
        (r'</TEXT>', ''),
        (r'<FILENAME>.*?</FILENAME>', ''),
        (r'<DESCRIPTION>.*?</DESCRIPTION>', ''),
        (r'<TYPE>.*?</TYPE>', ''),
        (r'<SEQUENCE>.*?</SEQUENCE>', ''),
        (r'<FILED>.*?</FILED>', ''),
        (r'<ACCESSION-NUMBER>.*?</ACCESSION-NUMBER>', ''),
        (r'<CONFORMED-SUBMISSION-TYPE>.*?</CONFORMED-SUBMISSION-TYPE>', ''),
        (r'<PUBLIC-DOCUMENT-COUNT>.*?</PUBLIC-DOCUMENT-COUNT>', ''),
        (r'<CONFORMED-PERIOD-OF-REPORT>.*?</CONFORMED-PERIOD-OF-REPORT>', ''),
        (r'<FILED-AS-OF-DATE>.*?</FILED-AS-OF-DATE>', ''),
        (r'<DATE-AS-OF-CHANGE>.*?</DATE-AS-OF-CHANGE>', ''),
        (r'<FILER>.*?</FILER>', ''),
        (r'<COMPANY-DATA>.*?</COMPANY-DATA>', ''),
        (r'<FILING-VALUES>.*?</FILING-VALUES>', ''),
        (r'<BUSINESS-ADDRESS>.*?</BUSINESS-ADDRESS>', ''),
        (r'<MAIL-ADDRESS>.*?</MAIL-ADDRESS>', ''),
        (r'<FORM-TYPE>.*?</FORM-TYPE>', ''),
        (r'<SEC-ACT>.*?</SEC-ACT>', ''),
        (r'<SEC-FILE-NUMBER>.*?</SEC-FILE-NUMBER>', ''),
        (r'<FILM-NUMBER>.*?</FILM-NUMBER>', ''),
        (r'<STREET1>.*?</STREET1>', ''),
        (r'<CITY>.*?</CITY>', ''),
        (r'<STATE>.*?</STATE>', ''),
        (r'<ZIP>.*?</ZIP>', ''),
        (r'<BUSINESS-PHONE>.*?</BUSINESS-PHONE>', ''),
        (r'<CENTRAL-INDEX-KEY>.*?</CENTRAL-INDEX-KEY>', ''),
        (r'<STANDARD-INDUSTRIAL-CLASSIFICATION>.*?</STANDARD-INDUSTRIAL-CLASSIFICATION>', ''),
        (r'<IRS-NUMBER>.*?</IRS-NUMBER>', ''),
        (r'<STATE-OF-INCORPORATION>.*?</STATE-OF-INCORPORATION>', ''),
        (r'<FISCAL-YEAR-END>.*?</FISCAL-YEAR-END>', '')
    ]
    
    for pattern, replacement in patterns:
        text = re.sub(pattern, replacement, text, flags=re.DOTALL)
    
    print(f"移除所有头部标记后长度: {len(text)}")
    
    # 移除XML声明和DOCTYPE
    text = re.sub(r'<\?xml.*?\?>', '', text, flags=re.DOTALL)
    text = re.sub(r'<!DOCTYPE.*?>', '', text, flags=re.DOTALL)
    
    # 移除空行
    text = re.sub(r'\n\s*\n', '\n', text)
    
    print("=== SEC文件头部处理完成 ===\n")
    return text

def clean_html_tags(text):
    """移除HTML标签"""
    print("\n=== 开始清理HTML标签 ===")
    print(f"输入文本长度: {len(text)}")
    
    # 移除所有HTML标签，但保留内容
    clean = re.compile('<.*?>')
    text = re.sub(clean, ' ', text)
    print(f"移除HTML标签后长度: {len(text)}")
    
    # 移除特定的HTML实体
    html_entities = {
        '&nbsp;': ' ',
        '&amp;': '&',
        '&lt;': '<',
        '&gt;': '>',
        '&quot;': '"',
        '&#x2018;': "'",
        '&#x2019;': "'",
        '&#x201C;': '"',
        '&#x201D;': '"',
        '&#x2013;': '-',
        '&#x2014;': '-'
    }
    
    for entity, replacement in html_entities.items():
        text = text.replace(entity, replacement)
    
    print(f"处理HTML实体后长度: {len(text)}")
    print("=== HTML标签清理完成 ===\n")
    return text

def clean_special_chars(text):
    """清理特殊字符"""
    print("\n=== 开始清理特殊字符 ===")
    print(f"输入文本长度: {len(text)}")
    
    # 替换特殊Unicode字符
    text = re.sub(r'[\u2018\u2019]', "'", text)
    text = re.sub(r'[\u201C\u201D]', '"', text)
    text = re.sub(r'[\u2013\u2014]', '-', text)
    print(f"替换Unicode字符后长度: {len(text)}")
    
    # 移除GAAP相关标签
    text = re.sub(r'us-gaap:[a-z]+', ' ', text)
    text = re.sub(r'dei:[a-z]+', ' ', text)
    text = re.sub(r'xbrli:[a-z]+', ' ', text)
    text = re.sub(r'gh:series[a-z]+', ' ', text)
    text = re.sub(r'gh-', ' ', text)
    print(f"移除GAAP标签后长度: {len(text)}")
    
    # 移除数字和特殊字符的组合，但保留纯文本
    text = re.sub(r'\d+[a-zA-Z]+', ' ', text)
    text = re.sub(r'[a-zA-Z]+\d+', ' ', text)
    print(f"移除数字字母组合后长度: {len(text)}")
    
    # 移除连续的数字，但保留单个数字
    text = re.sub(r'\d{2,}', ' ', text)
    print(f"移除连续数字后长度: {len(text)}")
    
    # 移除常见的无意义标签
    meaningless_tags = [
        r'<[^>]*>',  # 所有HTML标签
        r'&[a-z]+;',  # HTML实体
        r'[A-Z]{2,}(?::[A-Z]+)?',  # 大写字母标签
        r'[a-z]+-[a-z]+(?::[a-z]+)?',  # 连字符标签
        r':[a-z]+',  # 冒号开头的标签
        r'\.htm',  # .htm后缀
        r'onvertible',  # 拼写错误的convertible
    ]
    
    for pattern in meaningless_tags:
        text = re.sub(pattern, ' ', text)
    
    print(f"移除无意义标签后长度: {len(text)}")
    print("=== 特殊字符清理完成 ===\n")
    return text

def remove_repetitive_content(text):
    """移除重复内容"""
    print("\n=== 开始移除重复内容 ===")
    print(f"输入文本长度: {len(text)}")
    
    # 移除重复的单词序列
    words = text.split()
    unique_words = []
    prev_word = None
    prev_prev_word = None
    
    for word in words:
        # 跳过连续重复的单词
        if word == prev_word or word == prev_prev_word:
            continue
        unique_words.append(word)
        prev_prev_word = prev_word
        prev_word = word
    
    text = ' '.join(unique_words)
    print(f"移除重复内容后长度: {len(text)}")
    
    # 移除常见的无意义短语
    meaningless_phrases = [
        r'document\s+false',
        r'htm\s+document',
        r'gh:series[abcd]convertiblepreferredstockmember',
        r'gh:series[abcd]',
        r'convertiblepreferredstockmember',
        r'preferredstockmember',
        r'stockmember',
        r'member',
        r'us-gaap',
        r'dei:',
        r'xbrli:',
        r'accounting\s+standards\s+update',
        r'retained\s+earnings',
        r'accumulated\s+other\s+comprehensive\s+income',
        r'common\s+stock',
        r'additional\s+paid\s+in\s+capital',
        r'gh-',
        r'onvertible',
        r':accountingstandardsupdate',
        r':retainedearnings',
        r':accumulatedothercom'
    ]
    
    for phrase in meaningless_phrases:
        text = re.sub(phrase, '', text, flags=re.IGNORECASE)
    
    print(f"移除无意义短语后长度: {len(text)}")
    print("=== 重复内容移除完成 ===\n")
    return text

def normalize_text(text):
    """标准化文本"""
    print("\n=== 开始文本标准化 ===")
    print(f"原始文本长度: {len(text)}")
    
    # 移除SEC文件头部
    text = remove_sec_header(text)
    print(f"移除SEC头部后长度: {len(text)}")
    
    # 移除HTML标签
    text = clean_html_tags(text)
    print(f"清理HTML后长度: {len(text)}")
    
    # 清理特殊字符
    text = clean_special_chars(text)
    print(f"清理特殊字符后长度: {len(text)}")
    
    # 转换为小写
    text = text.lower()
    
    # 移除多余空白
    text = re.sub(r'\s+', ' ', text)
    print(f"清理空白后长度: {len(text)}")
    
    # 移除重复内容
    text = remove_repetitive_content(text)
    print(f"移除重复内容后长度: {len(text)}")
    
    # 移除短词（通常是无意义的），但保留一些重要的短词
    words = text.split()
    important_short_words = {'in', 'on', 'at', 'to', 'of', 'for', 'the', 'and', 'or', 'but', 'if', 'is', 'are', 'was', 'were', 'be', 'been', 'being'}
    words = [word for word in words if len(word) > 2 or word in important_short_words]
    text = ' '.join(words)
    print(f"移除短词后长度: {len(text)}")
    
    # 打印一些示例文本
    print("\n清理后的文本示例（前200个字符）:")
    print(text[:200])
    
    print("=== 文本标准化完成 ===\n")
    return text.strip()

def download_and_analyze(ticker):
    all_results = []
    for report in REPORT_TYPES:
        print(f"\n开始处理 {ticker} 的 {report} 报告...")
        try:
            downloader.get(report, ticker, after=f"{DATE_RANGE[0]}-01-01", before=f"{DATE_RANGE[1]}-12-31")
            filings_dir = os.path.join(OUTPUT_DIR, "sec-edgar-filings", ticker, report)
            if not os.path.exists(filings_dir):
                print(f"警告: 目录未找到: {filings_dir}")
                continue
                
            print(f"处理目录: {filings_dir}")
            # 使用 glob 递归搜索所有 .txt 文件
            txt_files = glob.glob(os.path.join(filings_dir, "**", "*.txt"), recursive=True)
            for filepath in txt_files:
                print(f"\n处理文件: {filepath}")
                with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read()
                
                print(f"原始文件大小: {len(text)} 字符")
                
                # 清理和标准化文本
                cleaned_text = normalize_text(text)
                print(f"清理后文本大小: {len(cleaned_text)} 字符")
                
                if len(cleaned_text) < 100:  # 如果清理后的文本太短，可能是清理过度
                    print("警告: 清理后的文本太短，使用原始文本")
                    cleaned_text = text
                
                # 运行情感分析
                scores = lm_classifier.predict(cleaned_text)
                print(f"\n{ticker} {report} {os.path.basename(filepath)} 的情感分析结果:")
                for category, score in scores.items():
                    print(f"  {category}: {score:.4f}")
                
                result = {
                    "ticker": ticker,
                    "report_type": report,
                    "file": filepath,
                    "positive": scores["positive"],
                    "negative": scores["negative"],
                    "uncertainty": scores["uncertainty"],
                    "litigious": scores["litigious"],
                    "constraining": scores["constraining"],
                    "superfluous": scores["superfluous"]
                }
                all_results.append(result)
        except Exception as e:
            print(f"处理 {ticker} {report} 时出错: {e}")
    return all_results

def main():
    final_results = []
    for sector, tickers in industry_groups.items():
        for ticker in tickers:
            results = download_and_analyze(ticker)
            for res in results:
                res["sector"] = sector
            final_results.extend(results)
    
    df = pd.DataFrame(final_results)
    output_csv = os.path.join(OUTPUT_DIR, "edgar_sentiment_scores.csv")
    df.to_csv(output_csv, index=False)
    print(f"Saved results to {output_csv}")

if __name__ == "__main__":
    main()
