import hashlib
import json
import time
from flask import Flask, request
from customize_service import mnist_service
from PIL import Image
import io
	 
app = Flask(__name__)
	 
model_name = "checkpoint_lenet_1-1_1875.ckpt"
model_path = "/home/ubuntu/infer/lenet/model"
# load model
# 初始化一个推理服务实例，实际模型文件放在 /home/ma-user/infer/lenet/model 路径下；
# 运行代码所在model目录在Dockerfile重新编译镜像 拷贝到容器 /home/ma-user/infer/lenet/model；
# 在模型导入时写 config.json 配置信息规范API的输入、输出参数，和代码要配合起来。
service_object = mnist_service(model_name, model_path)
	 
# 在线部署时填写的请求路径要和下面路由一致
@app.route('/infer/image', methods=['POST'])
def infer_image_func():
    print("----------- infer_image_func ----------")
    # 从config.json规定的请求体格式里面拿到文件数据
    file_data = request.files['images']
    # 很多时候，数据读写不一定是文件，也可以在内存中读写。StringIO就是在内存中读写str。BytesIO实现了在内存中读写bytes
    byte_stream = io.BytesIO(file_data.read())
    # 内部包装了一个含有原始数据的字典作为输入数据
    input_data = {"mnist":{"file1":byte_stream}}
    
    # 调用预处理函数
    preprocessed_result = service_object._preprocess(input_data)
    # 调用推理函数
    inference_result = service_object._inference(preprocessed_result)
    print("inference_result: %s", str(inference_result))
    # 调用后处理函数
    postprocess_result = service_object._postprocess(inference_result)
    print("infer_postprocess_result: %s", postprocess_result)
	
    # 从config.json规定的响应体格式组装输出的 "application/json" 格式数据
    mnist_result = []
    mnist_result.append(postprocess_result)
    res_data = {}
    res_data['mnist_result'] = mnist_result

    # 将 Python 对象编码成 JSON 字符串.缩进4个字符
    return json.dumps(res_data, indent=4)
    print("----------- infer_completed ----------")
	
# host must be "0.0.0.0", port must be 8080
if __name__ == '__main__':
    app.run(host="150.158.153.129", port=3306)
	
'''
         "request": {
            "Content-type": "multipart/form-data",
            "data": {
                "type": "object",
                "properties": {
					"images": {    
						"type": "array",
						"item": [{
							"type": "file"
                        }]
                    }
                }
            }
        }
'''
 
