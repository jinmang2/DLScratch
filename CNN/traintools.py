from utils import adjust_learning_rate


def train(model, criterion, optimizer, loaders, args):

    num_batches = len(trainLoader)
    trn_loss_list = []
    val_loss_list = []
    trainLoader, valLoader, testLoader = loaders
    device = args.device
    for epoch in range(args.num_epochs):
        adjust_learning_rate(optimizer, epoch)
        trn_loss = 0.0
        for i, data in enumerate(trainLoader):
            x, label = data[0].to(args.device), data[1].to(args.device)
            optimizer.zero_grad()
            model_output = model(x)
            loss = criterion(model_output, label)
            loss.backward()
            optimizer.step()
            
            trn_loss += loss.item()
            
            if (i+1) % args.verbose_epoch == 0:
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
                
                print("epoch: {}/{} | step: {}/{} | trn loss: {:.4f} | val loss: {:.4f} | acc: {:.2f}".format(
                    epoch+1, num_epochs, i+1, num_batches, trn_loss / 100, val_loss / len(valLoader), (corr_num / total_num) * 100
                ))            
                
                trn_loss_list.append(trn_loss/100)
                val_loss_list.append(val_loss/len(valLoader))
                trn_loss = 0.0

    print("training finished!")