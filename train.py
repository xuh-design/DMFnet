if __name__ == '__main__':
    from ultralytics import YOLO
    import ultralytics.nn.tasks
    import os
    
    # 正常训练
    # 获取当前文件所在目录的绝对路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, 'yaml', 'MPF_yolov8n.yaml')
    data_path = os.path.join(current_dir, 'data', 'drone2.yaml')
    
    model = YOLO(model_path)
    results = model.train(
        data=data_path,
        batch=4,
        epochs=300,
        device=0,
        imgsz=640,
        patience=300,
    )
