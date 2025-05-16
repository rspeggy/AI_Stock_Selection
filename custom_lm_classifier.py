import pandas as pd
import re
from collections import defaultdict
import os

class CustomLMClassifier:
    def __init__(self, dictionary_path=None):
        """
        初始化自定义的 Loughran-McDonald 情感分析分类器
        
        Args:
            dictionary_path: Loughran-McDonald 词典文件路径
        """
        # 定义词典文件中的列名映射
        self.category_mapping = {
            'positive': 'Positive',
            'negative': 'Negative',
            'uncertainty': 'Uncertainty',
            'litigious': 'Litigious',
            'constraining': 'Constraining',
            'superfluous': 'Complexity'  # 使用 Complexity 列作为 superfluous 的替代
        }
        
        self.categories = list(self.category_mapping.keys())
        self.word_lists = {category: set() for category in self.categories}
        
        if dictionary_path and os.path.exists(dictionary_path):
            print(f"Loading dictionary from: {dictionary_path}")
            self.load_dictionary(dictionary_path)
        else:
            print("Dictionary file not found, using default dictionary")
            self._initialize_default_dictionary()
    
    def _initialize_default_dictionary(self):
        """初始化默认词典"""
        # 这里添加一些基本的词典词条作为示例
        self.word_lists['positive'].update(['increase', 'growth', 'profit', 'success', 'gain'])
        self.word_lists['negative'].update(['decrease', 'loss', 'risk', 'failure', 'decline'])
        self.word_lists['uncertainty'].update(['may', 'might', 'possibly', 'uncertain', 'doubt'])
        self.word_lists['litigious'].update(['litigation', 'claim', 'sue', 'legal', 'court'])
        self.word_lists['constraining'].update(['must', 'shall', 'required', 'obligation', 'duty'])
        self.word_lists['superfluous'].update(['and', 'the', 'a', 'an', 'of'])
        
        # 打印默认词典的词数
        for category in self.categories:
            print(f"Default dictionary - {category}: {len(self.word_lists[category])} words")
    
    def load_dictionary(self, file_path):
        """
        从文件加载词典
        
        Args:
            file_path: 词典文件路径
        """
        try:
            print(f"Reading dictionary file: {file_path}")
            df = pd.read_csv(file_path)
            print(f"Dictionary columns: {df.columns.tolist()}")
            print("\nFirst few rows of dictionary:")
            print(df.head())
            
            # 获取所有单词
            all_words = df['Word'].str.lower().tolist()
            print(f"\nTotal words in dictionary: {len(all_words)}")
            
            # 更新类别映射
            self.category_mapping = {
                'positive': 'Positive',
                'negative': 'Negative',
                'uncertainty': 'Uncertainty',
                'litigious': 'Litigious',
                'constraining': 'Constraining',
                'superfluous': 'Complexity'  # 使用 Complexity 列作为 superfluous 的替代
            }
            
            # 清空现有的词列表
            self.word_lists = {category: set() for category in self.categories}
            
            # 加载每个类别的词
            for category, dict_column in self.category_mapping.items():
                if dict_column in df.columns:
                    # 获取该类别中非零值的单词
                    # 注意：这里我们假设任何非零值都表示该词属于该类别
                    words = df[df[dict_column] != 0]['Word'].str.lower().tolist()
                    self.word_lists[category].update(words)
                    print(f"Loaded {len(words)} words for category: {category}")
                    if len(words) > 0:
                        print(f"Sample words for {category}: {words[:5]}")
                else:
                    print(f"Warning: Column '{dict_column}' not found in dictionary")
            
            # 打印每个类别的词数
            for category in self.categories:
                print(f"Dictionary - {category}: {len(self.word_lists[category])} words")
                
            # 验证词列表是否为空
            empty_categories = [cat for cat, words in self.word_lists.items() if len(words) == 0]
            if empty_categories:
                print(f"Warning: The following categories have no words: {empty_categories}")
                print("Falling back to default dictionary")
                self._initialize_default_dictionary()
                
        except Exception as e:
            print(f"Error loading dictionary: {e}")
            print("Falling back to default dictionary")
            self._initialize_default_dictionary()
    
    def preprocess_text(self, text):
        """
        预处理文本
        
        Args:
            text: 输入文本
            
        Returns:
            处理后的文本列表
        """
        # 转换为小写
        text = text.lower()
        # 移除标点符号
        text = re.sub(r'[^\w\s]', ' ', text)
        # 分割成单词
        words = text.split()
        return words
    
    def count_words(self, words):
        """
        统计各类别词频
        
        Args:
            words: 预处理后的单词列表
            
        Returns:
            各类别词频统计
        """
        counts = defaultdict(int)
        total_words = len(words)
        
        for word in words:
            for category in self.categories:
                if word in self.word_lists[category]:
                    counts[category] += 1
        
        # 计算词频比例
        scores = {}
        for category in self.categories:
            scores[category] = counts[category] / total_words if total_words > 0 else 0
        
        return scores
    
    def predict(self, text):
        """
        对文本进行情感分析
        
        Args:
            text: 输入文本
            
        Returns:
            包含各类别情感得分的字典
        """
        words = self.preprocess_text(text)
        scores = self.count_words(words)
        return scores 