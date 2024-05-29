# 给yolo5增加api接口，使外部程序可以直接调用模型进行计算
直接运行CallApi.py即可进行测试，测试中的图像数据来自手机的`IP Camera`应用产生的http视频流。

## 文件修改
* 将原本`utils\dataloaders.py`文件里的`class LoadImages`改为`class LoadImages_ori_backup`起注释作用，防止被调用，并新增自定义的`class LoadImages`替代原函数的功能

## 文件增加
* `detect_api.py` 接口文件
* `CallApi.py` 调用实例

## 注
detect_api.py文件中的`non_max_suppression(pred, self.opt.conf_thres, self.opt.iou_thres, self.opt.classes, self.opt.agnostic_nms, max_det=20)`的max_det决定了最多绘制多少个目标框