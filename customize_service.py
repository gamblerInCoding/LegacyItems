import threading

import mindspore
import mindspore.nn as nn
import numpy as np
import logging
from mindspore import Tensor, context
from mindspore.common.initializer import Normal
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from model_service.model_service import SingleNodeService
from PIL import Image

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")

class LeNet5(nn.Cell):
    """Lenet network structure."""

    # define the operator required
    def __init__(self, num_class=10, num_channel=1):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(num_channel, 6, 5, pad_mode='valid')
        self.conv2 = nn.Conv2d(6, 16, 5, pad_mode='valid')
        self.fc1 = nn.Dense(16 * 5 * 5, 120, weight_init=Normal(0.02))
        self.fc2 = nn.Dense(120, 84, weight_init=Normal(0.02))
        self.fc3 = nn.Dense(84, num_class, weight_init=Normal(0.02))
        self.relu = nn.ReLU()
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()

    # use the preceding operators to construct networks
    def construct(self, x):
        x = self.max_pool2d(self.relu(self.conv1(x)))
        x = self.max_pool2d(self.relu(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class mnist_service(SingleNodeService):
    def __init__(self, model_name, model_path):
        self.model_name = model_name
        self.model_path = model_path
        logger.info("self.model_name:%s self.model_path: %s", self.model_name,
                    self.model_path)
        self.network = None
        # 非阻塞方式加载模型，防止阻塞超时
        thread = threading.Thread(target=self.load_model)
        thread.start()

    def load_model(self):
        logger.info("load network ... \n")
        self.network = LeNet5()
        ckpt_file = self.model_path + "/checkpoint_lenet_1-1_1875.ckpt"
        logger.info("ckpt_file: %s", ckpt_file)
        param_dict = load_checkpoint(ckpt_file)
        load_param_into_net(self.network, param_dict)
        # 模型预热，否则首次推理的时间会很长
        self.network_warmup()
        logger.info("load network successfully ! \n")

    def network_warmup(self):
        # 模型预热，否则首次推理的时间会很长
        logger.info("warmup network ... \n")
        images = np.array(np.random.randn(1, 1, 32, 32), dtype=np.float32)
        inputs = Tensor(images, mindspore.float32)
        inference_result = self.network(inputs)
        logger.info("warmup network successfully ! \n")

    def _preprocess(self, input_data):
        preprocessed_result = {}
        images = []
        for k, v in input_data.items():
            for file_name, file_content in v.items():
                image1 = Image.open(file_content)
                image1 = image1.resize((1, 32 * 32))
                image1 = np.array(image1, dtype=np.float32)
                images.append(image1)

        images = np.array(images, dtype=np.float32)
        logger.info(images.shape)
        images.resize([len(input_data), 1, 32, 32])
        logger.info("images shape: %s", images.shape)
        inputs = Tensor(images, mindspore.float32)
        preprocessed_result['images'] = inputs

        return preprocessed_result

    def _inference(self, preprocessed_result):
        inference_result = self.network(preprocessed_result['images'])
        return inference_result

    def _postprocess(self, inference_result):
        return str(inference_result)
