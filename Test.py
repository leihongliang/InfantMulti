# %%
frame_count = 50
EXTRACT_FREQUENCY = 4
if frame_count // EXTRACT_FREQUENCY <= 16:
    EXTRACT_FREQUENCY -= 1
    if frame_count // EXTRACT_FREQUENCY <= 16:
        EXTRACT_FREQUENCY -= 1
        if frame_count // EXTRACT_FREQUENCY <= 16:
            EXTRACT_FREQUENCY -= 1
print(EXTRACT_FREQUENCY)

# %%
import torch

m = torch.nn.AvgPool3d(kernel_size=3, stride = 1, padding=2, ceil_mode=True)
x = torch.ones(1, 2, 14, 14)
m(x).size()
# %%
import os
res = os.path.exists('/student1/hl_lei/InfantMulti/infant')
res2 = os.path.exists('./infant')
print(res,res2)
# %%
epoch_loss = 0.24324234
epoch_acc = 0.24324234
epoch_cp = 0.24324234
with open('/student1/hl_lei/InfantMulti/run/train.txt', 'w') as f:
    f.close()
with open('/train.txt', 'a') as f:
    f.write('%.5f\n' % (epoch_loss, epoch_acc, epoch_cp))
    f.close()