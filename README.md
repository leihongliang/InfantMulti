/run/run6
## ResNetXt run3
1cropsize = 112

## ResNetXt run6
1. cropsize = 224
2. scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)
3. optimizer = optim.SGD(train_params, lr=lr, momentum=0.9, weight_decay=5e-4)
4. lr = 5e-3 

## ResNet run8
1. cropsize = 112
2. scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)
3. optimizer = optim.SGD(train_params, lr=lr, momentum=0.9, weight_decay=5e-4)
4. lr = 5e-3 

## ResNet run9
1. cropsize = 112
2. scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.8)
3. optimizer = optim.SGD(train_params, lr=lr, momentum=0.9, weight_decay=5e-4)
4. lr = 5e-3 