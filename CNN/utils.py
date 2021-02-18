# learning rate를 단계적으로 줄여주는 방법
# epoch 100 -> lr/10, 150 -> lr/10
def adjust_learning_rate(optimizer, epoch, lr):
    if epoch >= 100:
        lr /= 10
    if epoch >= 150:
        lr /= 10
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
