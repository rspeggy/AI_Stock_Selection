import qlib
from qlib.config import REG_CN
from qlib.data import D
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord

def init_qlib(provider_uri="~/.qlib/qlib_data/cn_data"):
    """
    初始化Qlib环境
    
    Args:
        provider_uri (str): 数据提供者URI路径
    """
    try:
        qlib.init(provider_uri=provider_uri, region=REG_CN)
        print("Qlib初始化成功")
    except Exception as e:
        print(f"Qlib初始化失败: {str(e)}")

def get_stock_data(stock_code, start_time, end_time, fields=None):
    """
    获取股票数据
    
    Args:
        stock_code (str): 股票代码
        start_time (str): 开始时间
        end_time (str): 结束时间
        fields (list): 需要获取的字段列表
    
    Returns:
        pd.DataFrame: 股票数据
    """
    if fields is None:
        fields = ["$close", "$volume", "$factor", "$high", "$low", "$open"]
    
    try:
        data = D.features([stock_code], fields, start_time=start_time, end_time=end_time)
        return data
    except Exception as e:
        print(f"获取股票数据失败: {str(e)}")
        return None

def save_model_record(model, model_name, dataset, experiment_name):
    """
    保存模型记录
    
    Args:
        model: 训练好的模型
        model_name (str): 模型名称
        dataset: 数据集
        experiment_name (str): 实验名称
    """
    try:
        with R.start(experiment_name=experiment_name):
            R.log_params(model_name=model_name)
            R.log_metrics(train_score=model.score(dataset))
            R.save_objects(model=model)
        print(f"模型记录保存成功: {model_name}")
    except Exception as e:
        print(f"保存模型记录失败: {str(e)}")

if __name__ == "__main__":
    # 示例用法
    init_qlib()
    stock_code = "000001.SZ"  # 示例股票代码
    start_time = "2020-01-01"
    end_time = "2021-01-01"
    
    data = get_stock_data(stock_code, start_time, end_time)
    if data is not None:
        print(data.head()) 