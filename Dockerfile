# 基础镜像：Python 3.11
FROM python:3.11-slim

# 设置工作目录
WORKDIR /app

# 先复制依赖文件（利用缓存，依赖没变就不重新安装）
COPY requirements.txt .

# 安装依赖
RUN pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 复制项目代码
COPY . .

# 暴露端口
EXPOSE 8000

# 启动命令
CMD ["uvicorn", "examples.api:app", "--host", "0.0.0.0", "--port", "8000"]