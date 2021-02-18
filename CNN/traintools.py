import torch
from tqdm.auto import tqdm
from utils import adjust_learning_rate


def train(model, criterion, optimizer, loaders, args):
    trn_loss_list = []
    val_loss_list = []
    trainLoader, valLoader, testLoader = loaders
    num_batches = len(trainLoader)
    device = args.device
    for epoch in tqdm(range(args.num_epochs)):
        adjust_learning_rate(optimizer, epoch, args.learning_rate)
        trn_loss = 0.0
        for i, data in enumerate(trainLoader):
            x, label = data[0].to(args.device), data[1].to(args.device)
            optimizer.zero_grad()
            model_output = model(x)
            loss = criterion(model_output, label)
            loss.backward()
            optimizer.step()

            trn_loss += loss.item()

            if i % (num_batches // 2) == 0:
                with torch.no_grad():
                    val_loss = 0.0
                    corr_num, total_num = 0, 0
                    for j, val in enumerate(valLoader):
                        val_x, val_label = val[0].to(device), val[1].to(device)
                        val_output = model(val_x)
                        v_loss = criterion(val_output, val_label)
                        val_loss += v_loss

                        model_label = val_output.argmax(dim=1)
                        corr = val_label[val_label == model_label].size(0)
                        corr_num += corr
                        total_num += val_label.size(0)

                print(f"epoch: {epoch+1:03d}/{args.num_epochs} | "
                      f"step: {i+1:03d}/{num_batches} | "
                      f"trn loss: {trn_loss/100:.4f} "
                      f"| val loss: {val_loss/len(valLoader):08.4f} "
                      f"| acc: {(corr_num/total_num)*100:06.2f}")

                trn_loss_list.append(trn_loss/100)
                val_loss_list.append(val_loss/len(valLoader))
                trn_loss = 0.0

    print("training finished!")
