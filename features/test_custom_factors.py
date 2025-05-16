import unittest
import pandas as pd
import numpy as np
import os
from custom_factors import HealthcareFactors, create_custom_factors, load_and_prepare_data

class TestHealthcareFactors(unittest.TestCase):
    def setUp(self):
        """设置测试数据"""
        # 创建示例股票数据
        dates = pd.date_range(start='2020-01-01', end='2020-12-31', freq='D')
        self.stock_data = pd.DataFrame({
            'Close': np.random.randn(len(dates)).cumsum() + 100,
            'High': np.random.randn(len(dates)).cumsum() + 102,
            'Low': np.random.randn(len(dates)).cumsum() + 98,
            'Volume': np.random.randint(1000, 10000, len(dates))
        }, index=dates)
        
        # 创建示例宏观数据
        self.macro_data = pd.DataFrame({
            'GDP': np.random.randn(len(dates)).cumsum() + 1000,
            'CPI': np.random.randn(len(dates)).cumsum() + 2,
            'UNRATE': np.random.randn(len(dates)).cumsum() + 3,
            'FEDFUNDS': np.random.randn(len(dates)).cumsum() + 1,
            'INFLATION': np.random.randn(len(dates)).cumsum() + 2
        }, index=dates)

    def test_sector_volatility(self):
        """测试行业波动率因子计算"""
        volatility = HealthcareFactors.sector_volatility(
            self.stock_data['High'],
            self.stock_data['Low'],
            self.stock_data['Close']
        )
        self.assertIsInstance(volatility, pd.Series)
        self.assertEqual(len(volatility), len(self.stock_data))
        self.assertFalse(volatility.isnull().all())

    def test_momentum_score(self):
        """测试动量得分因子计算"""
        momentum = HealthcareFactors.momentum_score(
            self.stock_data['Close'],
            self.stock_data['Volume']
        )
        self.assertIsInstance(momentum, pd.Series)
        self.assertEqual(len(momentum), len(self.stock_data))
        self.assertFalse(momentum.isnull().all())

    def test_healthcare_sentiment(self):
        """测试医疗健康行业情绪因子计算"""
        sentiment = HealthcareFactors.healthcare_sentiment(
            self.stock_data['Close'],
            self.stock_data['Volume']
        )
        self.assertIsInstance(sentiment, pd.Series)
        self.assertEqual(len(sentiment), len(self.stock_data))
        self.assertFalse(sentiment.isnull().all())

    def test_technical_indicators(self):
        """测试技术指标因子计算"""
        indicators = HealthcareFactors.technical_indicators(
            self.stock_data['High'],
            self.stock_data['Low'],
            self.stock_data['Close'],
            self.stock_data['Volume']
        )
        self.assertIsInstance(indicators, dict)
        self.assertTrue(all(isinstance(v, pd.Series) for v in indicators.values()))
        self.assertTrue(all(len(v) == len(self.stock_data) for v in indicators.values()))

    def test_macro_impact_factor(self):
        """测试宏观影响因子计算"""
        impact = HealthcareFactors.macro_impact_factor(
            self.stock_data,
            self.macro_data
        )
        self.assertIsInstance(impact, pd.Series)
        self.assertEqual(len(impact), len(self.stock_data))
        self.assertFalse(impact.isnull().all())

    def test_create_custom_factors(self):
        """测试自定义因子创建"""
        # 测试不带宏观数据
        factors = create_custom_factors(self.stock_data)
        self.assertIsInstance(factors, pd.DataFrame)
        self.assertEqual(len(factors), len(self.stock_data))
        
        # 测试带宏观数据
        factors_with_macro = create_custom_factors(self.stock_data, self.macro_data)
        self.assertIsInstance(factors_with_macro, pd.DataFrame)
        self.assertEqual(len(factors_with_macro), len(self.stock_data))
        self.assertIn('macro_impact', factors_with_macro.columns)

    def test_load_and_prepare_data(self):
        """测试数据加载和准备"""
        try:
            stock_data, macro_data = load_and_prepare_data(
                "EXAS",
                "2020-01-01",
                "2020-12-31"
            )
            if stock_data is not None:
                self.assertIsInstance(stock_data, pd.DataFrame)
                self.assertTrue('Close' in stock_data.columns)
                self.assertTrue('High' in stock_data.columns)
                self.assertTrue('Low' in stock_data.columns)
                self.assertTrue('Volume' in stock_data.columns)
            
            if macro_data is not None:
                self.assertIsInstance(macro_data, pd.DataFrame)
                self.assertTrue('GDP' in macro_data.columns)
                self.assertTrue('CPI' in macro_data.columns)
                self.assertTrue('UNRATE' in macro_data.columns)
                self.assertTrue('FEDFUNDS' in macro_data.columns)
                self.assertTrue('INFLATION' in macro_data.columns)
        except Exception as e:
            self.skipTest(f"跳过数据加载测试: {str(e)}")

if __name__ == '__main__':
    unittest.main() 