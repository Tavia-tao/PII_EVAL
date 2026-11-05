import os

# 测试只连接显式的协议桩，不启用第三方遥测或更新检查。
os.environ["DEEPEVAL_TELEMETRY_OPT_OUT"] = "YES"
os.environ["DEEPEVAL_UPDATE_WARNING_OPT_OUT"] = "YES"
