
modelName = '3d'
nEpochs = 400
lr = 5e-3

with open('/test/train.txt', 'w') as f:
    f.write("modelName" + modelName + ("\n") +
            "nEpochs: " + nEpochs + ("\n") +
            "lr: " + lr + ("\n") +
            "nEpochs: " + nEpochs + ("\n") +
            "scheduler: " + scheduler.step() + ("\n") +
            "resize_height: {} resize_width: {} crop_size: {}".format(VideoDataset.self.resize_height,
                                                                      VideoDataset.self.resize_width,
                                                                      VideoDataset.self.crop_size) + ("\n"))
    f.close()
